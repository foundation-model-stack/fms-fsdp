import math
import os
from pathlib import Path

import fire
import torch
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecisionPolicy
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.pipelining import ScheduleGPipe
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

from torch.distributed.device_mesh import init_device_mesh


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

    mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("pp", "dp"))
    pp_mesh = mesh["pp"]
    dp_mesh = mesh["dp"]
    print("mesh: ", mesh)
    print("pp mesh: ", pp_mesh)
    print("dp mesh: ", dp_mesh)

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

    # PP
    pp_rank = pp_mesh.get_local_rank()
    print(f"rank: {rank}, pp rank: {pp_rank}")
    pp_size = pp_mesh.size()

    stage_index = pp_rank
    num_stages = pp_size


    num_layers = config_data["n_layer"]
    # stage_layer_map = {
    #     0: [0, 1],
    #     1: [2, 3],
    #     2: [4, 5],
    #     3: [6, 7],
    #     4: [8, 9],
    #     5: [10, 11],
    #     6: [12, 13],
    #     7: [14, 15],
    #     8: [16, 17],
    #     9: [18, 19],
    #     10: [20, 21],
    #     11: [22, 23],
    #     12: [24, 25],
    #     13: [26, 27],
    #     14: [28, 29],
    #     15: [30, 31],
    # }

    stage_layer_map = {
        0: range(0, 16),
        1: range(16, 32)
    }

    if stage_index != 0:
        model.backbone.embedding = None
    if stage_index != num_stages - 1:
        model.backbone.norm_f = None
        model.lm_head = None
    for i in range(num_layers):
        if i not in stage_layer_map[stage_index]:
            del model.backbone.layers[str(i)]


    for layer_index, block in model.backbone.layers.items():
        model.backbone.layers[layer_index] = checkpoint_wrapper(block, preserve_rng_state=False)


    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for layer_index, block in model.backbone.layers.items():
        fully_shard(block, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)




    print("stage index: ", stage_index, model)
    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        torch.device(f"cuda:{local_rank}"),
        group=pp_mesh.get_group("pp"),
    )

    def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            pred.view(-1, pred.size(-1)), labels.view(-1).long()
        )

    pp_schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=cross_entropy_loss)


    # # fsdp activation checkpointing
    # if cfg.fsdp_activation_checkpointing:
    #     if rank == 0:
    #         print(f"--> applying FSDP activation checkpointing...")
    #     apply_selective_ac(model, p=cfg.selective_checkpointing)

    # # torch compile
    # if cfg.use_torch_compile:
    #     if rank == 0:
    #         print(f"--> enabling torch compile...")
    #     # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
    #     torch._dynamo.config.accumulated_cache_size_limit = 128
    #     model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
        model,
        optimizer,
        None,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
        if not os.path.isfile(cfg.ckpt_load_path)
        else cfg.ckpt_load_path,
        strict=False,
    )
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # LR schedule
    # linear decay for annealing
    if cfg.training_stage == "annealing":
        warmup_interval = 1000
        schedule = lambda x: x / warmup_interval if x < warmup_interval else 1 - (x - warmup_interval) / (cfg.num_steps - warmup_interval)
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
        pp_schedule,
        stage_index,
        num_stages,
    )

    checkpointer.save_single_file(cfg.num_steps, model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
