import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMA
from fms.models import get_model
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.optim.lr_scheduler import LambdaLR

from train_speculator_utils import train_speculator_stage1

from fms_fsdp import config, policies
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
)

from fms_extras.models import MLPSpeculator


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)
    base_seq_len = cfg.seq_length
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
    mixed_precision_policy, wrapping_policy, sharding_strategy_policy = get_policies(
        cfg, rank
    )

    # get base model
    model = get_model(
        "llama",
        "7b",
        model_path = cfg.model_path,
        device_type = "cuda",
        source = "hf",
        distributed_strategy = sharding_strategy_policy
    )

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
        total_params = sum(p.numel() for p in speculator.parameters() if p.requires_grad)
        print(f"\n--> speculator has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
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
        speculator.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    speculator, optimizer, train_loader, start_step, tokens_seen = checkpointer.load(
        speculator,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
    )

    # LR schedule
    warmup_interval = min(2000, cfg.num_steps // 20)
    schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
        0.1
        + 0.5
        * (1 - 0.1)
        * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train_speculator_stage1(
        cfg,
        model,
        speculator,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
