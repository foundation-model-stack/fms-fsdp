import math
import os
from pathlib import Path

import fire
import torch
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Block
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

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
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    # get policy
    block = Block
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block)

    # get fms model
    if cfg.model_variant == "mamba_1.5b":
        config_data = {
            "d_model": 2048,
            "n_layer": 48,
            "vocab_size": cfg.vocab_size,
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 8,
        }
    elif cfg.model_variant == "mamba_7b":
        config_data = {
            "d_model": 4096,
            "n_layer": 64,
            "vocab_size": cfg.vocab_size,
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 8,
        }
    else:
        raise ValueError(f"model variant {cfg.model_variant} not supported.")
    mamba_config = MambaConfig(**config_data)
    if cfg.low_cpu_fsdp:
        raise ValueError(
            "For unknown reason, low_cpu_mode will make Mamba model hangs. Please disable low_cpu_mode."
        )
    else:
        model = MambaLMHeadModel(mamba_config)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

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
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print(f"--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    model, optimizer, train_loader, start_step, tokens_seen = checkpointer.load(
        model,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
    )

    # LR schedule
    warmup_interval = min(2000, cfg.num_steps // 20)
    decay_factor = 1e-5 / cfg.learning_rate
    schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
        decay_factor
        + 0.5
        * (1 - decay_factor)
        * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        model,
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
