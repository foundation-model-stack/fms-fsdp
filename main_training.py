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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.pipelining import Schedule1F1B
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
    num_layers = config_data.n_layer
    num_stages = world_size

    stage_layer_map = {
        0: [0],
        1: [1, 2],
        2: [3, 4],
        3: [5, 6],
        4: [7, 8],
        5: [9, 10],
        6: [11, 12],
        7: [13, 14],
        8: [15, 16],
        9: [17, 18],
        10: [19, 20],
        11: [21, 22],
        12: [23, 24],
        13: [25, 26],
        14: [27, 28],
        15: [29, 30],
        16: [31],
    }
    stage_index = rank
    if stage_index != 0:
        model.backbone.embedding = None
    elif stage_index != num_stages - 1:
        model.backbone.norm_f = None
        model.lm_head = None
    for i in range(num_layers):
        if i not in stage_layer_map[stage_index]:
            del model.backbone.layers[str(i)]

    print(stage_index)
    print(model)

    from torch.distributed.pipelining import PipelineStage
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        torch.cuda.current_device(),
    )

    def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = pred.logits if hasattr(pred, "logits") else pred
        ce_loss = torch.nn.CrossEntropyLoss()
        return ce_loss(
            pred.view(-1, pred.size(-1)), labels.view(-1).long()
        )

    pp_schedule = Schedule1F1B(stage, n_microbatches=1, loss_fn=cross_entropy_loss)


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
    if cfg.training_stage == "annealing":
        schedule = lambda x: 1 - x / cfg.num_steps
    else:
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
