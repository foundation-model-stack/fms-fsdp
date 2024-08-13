import math
import os

import fire  # type: ignore
import torch
import torch.optim as optim
from fms.utils import serialization
from fms.models import get_model, register_model
from fms.models.llama import LLaMABlock, LLaMAConfig
from fms.models.llama import _hf_sd_to_fms_sd as _llama_hf_sd_to_fms_sd
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
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
)
from speculator.train_speculator_utils import EmbedLLaMA, train_speculator


def _llama_factory_factory(config):
    def factory(**kwargs):
        return EmbedLLaMA(config, **kwargs)

    return factory


register_model("embedllama", "7b", _llama_factory_factory(LLaMAConfig()))
serialization.register_adapter("embedllama", "hf", _llama_hf_sd_to_fms_sd)


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
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, LLaMABlock)

    # get base model
    model = get_model(
        "embedllama",
        "7b",
        model_path=cfg.model_path,
        device_type="cuda",
        source="hf",
        distributed_strategy=cfg.sharding_strategy,
    )
    model = model.bfloat16()

    # get speculator
    speculator = MLPSpeculator(
        model.config.emb_dim,
        cfg.speculator_width,
        model.config.src_vocab_size,
        cfg.n_speculator_heads,
    )
    speculator.reset_parameters()

    # llama_config = get_model_config(cfg.model_variant)

    # if cfg.low_cpu_fsdp:
    #     if rank == 0:
    #         model = LLaMA(llama_config)
    #         model.reset_parameters()
    #     else:
    #         with torch.device("meta"):
    #             model = LLaMA(llama_config)
    # else:
    #     model = LLaMA(llama_config)
    #     model.reset_parameters()

    if rank == 0:
        total_params = sum(
            p.numel() for p in speculator.parameters() if p.requires_grad
        )
        print(f"\n--> speculator has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size, postprocess=[])
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

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
    )

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print(f"--> enabling torch compile...")
            if cfg.fsdp_activation_checkpointing:
                raise ValueError(
                    "Compile does not yet work well with llama+ac, please"
                    "either use it without activation checkpointing, or disable"
                    "compile."
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
    checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, "ddp", rank, local_rank)
    speculator, optimizer, train_loader, start_step, tokens_seen, _ = checkpointer.load(
        speculator,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
    )

    # LR schedule
    # These functions provide LR scaling factors in [0,1] based on step count.
    # Stage 1: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of max LR.
    warmup_interval1 = min(2000, cfg.stage2_start_step // 20)
    stage1_schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval1) / warmup_interval1) ** 2,
        0.1
        + 0.5
        * (1 - 0.1)
        * (
            1
            + math.cos(min(x, cfg.stage2_start_step) / cfg.stage2_start_step * math.pi)
        ),
    )
    # Stage 2: warm up over first 2k or 5% of steps, whichever is smaller.
    # Then cosine anneal to 10% of stage 1's final LR.
    warmup_interval2 = min(2000, (cfg.num_steps - cfg.stage2_start_step) // 20)
    stage2_schedule = lambda x: min(
        0.1 * (1 - (1 - min(x, warmup_interval2) / warmup_interval2) ** 2),
        0.01
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
    schedule = (
        lambda x: stage1_schedule(x)
        if x <= cfg.stage2_start_step
        else stage2_schedule(x - cfg.stage2_start_step)
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
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
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
