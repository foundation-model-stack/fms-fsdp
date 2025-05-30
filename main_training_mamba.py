import math
import os
from pathlib import Path

import fire
import torch
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp.utils.checkpoint import Checkpointer
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.model_configs import get_model_config, update_config
from fms_fsdp.utils.train_config import train_config
from fms_fsdp.utils.train_utils import get_profiler, setup, setup_environ_flags, train


def main(**kwargs):
    # get configs
    cfg = train_config()
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

    # get model
    config_data = get_model_config(cfg.model_variant)
    mamba_config = MambaConfig(**config_data)
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

    # AC
    if cfg.fsdp_activation_checkpointing:
        for layer_index, block in enumerate(model.backbone.layers):
            model.backbone.layers[layer_index] = checkpoint_wrapper(
                block, preserve_rng_state=False
            )

    # FSDP
    if cfg.sharding_strategy == "hsdp":
        mesh = init_device_mesh(
            "cuda", (world_size // 8, 8), mesh_dim_names=("dp_replicate", "dp_shard")
        )
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    for layer_index, block in enumerate(model.backbone.layers):
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=layer_index < len(model.backbone.layers) - 1,
        )
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)

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

    # Train State
    train_state = {"step": 0, "ntokens": 0}

    # Load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(cfg.ckpt_save_path)
    model, optimizer, train_state = checkpointer.load(
        model,
        optimizer,
        train_state,
    )
    start_step = train_state["step"]
    tokens_seen = train_state["ntokens"]

    # LR schedule
    # linear decay for annealing
    if cfg.training_stage == "annealing":
        warmup_interval = 1000
        schedule = (
            lambda x: x / warmup_interval
            if x < warmup_interval
            else 1 - (x - warmup_interval) / (cfg.num_steps - warmup_interval)
        )
    elif cfg.training_stage == "cosine":
        # cosine decay
        warmup_interval = min(2000, cfg.num_steps // 20)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )
    elif cfg.training_stage == "constant":
        warmup_interval = 2000
        schedule = lambda x: (min(x, warmup_interval) / warmup_interval)
    else:
        schedule = lambda x: 1.0 + (0.75 - 1.0) * (x / 32000) if x <= 32000 else 0.75

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
