import math
import os
import time

import fire
import torch
import torch.optim as optim
from fms.models import get_model
from fms.models.llama import LLaMABlock
from fms.utils import generation, tokenizers
from fms_extras.models.speculator import MLPSpeculator  # type: ignore
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_mixed_precision_policy,
    get_profiler,
    setup,
    setup_environ_flags,
)
from speculator.train_speculator_utils import train_speculator


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_model(rank, model, arch, cfg, prompt_type="chat"):
    if rank == 0:
        print("testing model output")
    tokenizer = tokenizers.get_tokenizer(cfg.model_path)
    if prompt_type == "chat":
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
        prompt = template.format(
            "Provide a list of instructions for preparing chicken soup."
        )
    else:
        template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
        prompt = template.format("Write a bubble sort function in python.")

    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if "llama" in arch:
        ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device="cuda")
    result = generation.generate(
        model,
        ids,
        max_new_tokens=100,
        use_cache=True,
        do_sample=False,
        max_seq_len=8192,
    )
    result = generation.truncate_after_eos(result, tokenizer.eos_token_id)
    if rank == 0:
        print(f"{rank}: quick test of base model")
        print(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
        )


def get_emb_dim(model):
    if hasattr(model.config, "emb_dim"):
        emb_dim = model.config.emb_dim
    elif hasattr(model.config, "dim"):  # Mixtral
        emb_dim = model.config.dim
    elif hasattr(model.config, "hidden_size"):  # HF
        emb_dim = model.config.hidden_size
    else:
        raise Exception("config missing embedding dimension")
    return emb_dim


def get_vocab_size(model):
    if hasattr(model.config, "src_vocab_size"):  # FMS
        vocab_size = model.config.src_vocab_size
    elif hasattr(model.config, "vocab_size"):  # HF
        vocab_size = model.config.vocab_size
    else:
        raise Exception("config missing vocab size config")
    return vocab_size


def get_training_data_loader(rank, cfg, world_size, speculator_mesh):
    if rank == 0:
        print(f"{time.time()} Constructing datasets...")
    if not cfg.use_dummy_dataset:
        if cfg.sharding_strategy == "tp":
            train_loader = get_data_loader(
                cfg, speculator_mesh.get_rank(), speculator_mesh.size(), postprocess=[]
            )
        else:
            train_loader = get_data_loader(cfg, rank, world_size, postprocess=[])
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print(f"{time.time()} Datasets constructed!")
    return train_loader


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)
    cfg.seq_length = cfg.seq_length + cfg.n_speculator_heads + 1

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"{time.time()} running with these configs {cfg}")

    # some setups
    torch.cuda.set_device(local_rank)

    if cfg.sharding_strategy != "tp":
        setup()
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
        base_model_mesh = None
        speculator_mesh = None
    else:
        base_model_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (world_size // cfg.tp_size, cfg.tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        speculator_mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
        torch._C._distributed_c10d._register_process_group(
            "default", base_model_mesh["tp"].get_group()
        )

    torch.cuda.empty_cache()
    setup_environ_flags()
    torch.set_default_dtype(torch.bfloat16)

    mixed_precision_policy = get_mixed_precision_policy(cfg, rank)

    model = get_model(
        cfg.model_arch,
        cfg.model_variant,
        model_path=cfg.model_path,
        device_type="cuda",
        source="hf",
        distributed_strategy=cfg.sharding_strategy,
        group=(
            base_model_mesh["tp"].get_group() if cfg.sharding_strategy == "tp" else None
        ),
    )

    if rank == 0:
        print(f"{time.time()}")
        print(model.config)
        print(model)

    model.eval()
    with torch.no_grad():
        test_model(rank, model, cfg.model_arch, cfg)

    emb_dim = get_emb_dim(model)
    vocab_size = get_vocab_size(model)

    # get speculator
    if rank == 0:
        print(f"{time.time()} Loading speculator")
    speculator = MLPSpeculator(
        emb_dim,
        cfg.speculator_width,
        vocab_size,
        cfg.n_speculator_heads,
        tie_weights=cfg.speculator_tie_weights,
        scale_input=cfg.speculator_scale_input,
    )
    speculator.reset_parameters()

    if rank == 0:
        total_params = sum(
            p.numel() for p in speculator.parameters() if p.requires_grad
        )
        print(f"\n{time.time()} speculator has {total_params / 1e6} Million params\n")

    # get data loader
    train_loader = get_training_data_loader(rank, cfg, world_size, speculator_mesh)

    # FSDP
    speculator = FSDP(
        speculator,
        auto_wrap_policy=None,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: (
            module.to_empty(device=torch.device("cuda"), recurse=False)
            if cfg.low_cpu_fsdp
            else None
        ),
        device_mesh=speculator_mesh if cfg.sharding_strategy == "tp" else None,
    )

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print(f"enabling torch compile...")
            if cfg.fsdp_activation_checkpointing:
                raise ValueError(
                    "Compile does not yet work well with llama+ac, please"
                    "either use it without activation checkpointing, or disable"
                    "compile."
                )
        # we need this post-fsdp call to avoid graph break with torch.compile,
        if cfg.sharding_strategy != "tp" and hasattr(model, "rot_emb"):
            model.rot_emb.compute_freqs_cis(
                torch.device("cuda", torch.cuda.current_device()),
                model.config.max_expected_seq_len + 10,
            )
        model = torch.compile(model)
        speculator = torch.compile(speculator)

    # Optimizer
    optimizer = optim.AdamW(
        speculator.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # optionally load from checkpoint (when continue pretraining)
    if cfg.sharding_strategy == "tp":
        checkpointer = Checkpointer(
            cfg.ckpt_save_path,
            1000,
            "ddp",
            speculator_mesh.get_rank(),
            speculator_mesh.get_local_rank(),
            model_auto_placement=True,
        )
    else:
        checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, "ddp", rank, local_rank)
    speculator, optimizer, train_loader, start_step, tokens_seen, _ = checkpointer.load(
        speculator,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
        is_compiled=cfg.use_torch_compile,
    )

    # LR schedule
    # These functions map step count to LR scaling factor in [0,1].
    # Stage 1: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of max LR.
    warmup_interval1 = min(2000, cfg.stage2_start_step // 20)
    stage1_schedule = lambda x: min(
        # Parabolic warmup
        1 - (1 - min(x, warmup_interval1) / warmup_interval1) ** 2,
        # Final .1 scaling factor
        0.1
        # Cosine anneal from 1 to .1 over stage2_start_step steps
        + 0.5 * (1 - 0.1) * (1 + math.cos(x / cfg.stage2_start_step * math.pi)),
    )
    # Stage 2: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of stage 1's final LR.
    warmup_interval2 = min(2000, (cfg.num_steps - cfg.stage2_start_step) // 20)
    stage2_schedule = lambda x: min(
        # Parabolic warmup to stage2's max LR (10% of stage1's max LR)
        0.1 * (1 - (1 - min(x, warmup_interval2) / warmup_interval2) ** 2),
        # Final 10% of 10% scaling factor
        0.01
        # Cosine anneal from .1 to .01 over remaining stage2 steps
        + 0.05
        * (1 - 0.1)
        * (
            1
            + math.cos(
                min(x, cfg.num_steps - cfg.stage2_start_step)
                / (cfg.num_steps - cfg.stage2_start_step)
                * math.pi
            )
        ),
    )
    # Assemble full scheduling function with correct step offsets.
    schedule = lambda x: (
        stage1_schedule(x)
        if x <= cfg.stage2_start_step
        else stage2_schedule(x - cfg.stage2_start_step)
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"{time.time()} Training for {cfg.num_steps} steps")
    torch.cuda.empty_cache()
    train_speculator(
        cfg,
        model,
        speculator,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        start_step,
        tokens_seen,
        profiler,
        base_model_mesh,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
