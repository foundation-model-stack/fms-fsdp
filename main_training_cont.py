"""
Version to train a loaded-in checkpoint on a new dataset
"""

import math
import os,sys

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMA, LLaMABlock
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

    print(f"\n** local_rank: {local_rank} == rank: {rank} == world_size: {world_size}") # test later with -n 2 in bsub

    if rank == 0:
        print(f"\n--> running with these configs {type(cfg)} {cfg}")
        '''
        --> running with these configs train_config(
        model_variant='llama2mod_starcoder', 
        ckpt_load_path='/proj/data-eng/fsdp/experiments/R83a/', 
        ckpt_save_path='/proj/data-eng/fsdp/experiments/R83b/', 
        use_dummy_dataset=False, 
        data_path='/proj/data-eng/fsdp/data/R83b', 
        datasets='CC-MAIN-2024-10,CC-MAIN-2023-40', 
        weights=(60.0, 40.0), 
        seq_length=8192, vocab_size=49152, 
        bos_token=None, eos_token=0, 
        bol_token=None, eol_token=None, 
        strip_tokens='', 
        logical_shards=640, 
        sharding_strategy='hsdp', 
        fsdp_activation_checkpointing=False, 
        selective_checkpointing=1, 
        mixed_precision=True, low_cpu_fsdp=False, batch_size=2, num_steps=5, learning_rate=0.0006, grad_clip_thresh=1.0, seed=42, use_profiler=False, profiler_rank0_only=True, report_interval=2, checkpoint_interval=3, tracker='aim', tracker_dir='/proj/data-eng/fsdp/data/R83b/aim', tracker_project_name='001', tracker_run_id=None, use_torch_compile=True)
        '''

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    block = LLaMABlock
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block)

    # get fms model
    llama_config = get_model_config(cfg.model_variant)
    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = LLaMA(llama_config)
    else:
        model = LLaMA(llama_config)
        model.reset_parameters()

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
    # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
    model.rot_emb.compute_freqs_cis(
        torch.device("cuda", torch.cuda.current_device()),
        model.config.max_expected_seq_len,
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

    

    model, optimizer, _, start_step, tokens_seen = checkpointer.load(
        model,
        optimizer,
        None,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/"),
    )

    print(f"\n==AFTER chkp loading: start_step: {start_step} tokens_seen: {tokens_seen}")

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
    
    # my_kwargs={'seed': 42, 'model_variant': 'llama2mod_starcoder', 'use_dummy_dataset': False, 'ckpt_load_path': '/proj/data-eng/fsdp/experiments/R83b/', 'ckpt_save_path': '/proj/data-eng/fsdp/experiments/R83b/', 'selective_checkpointing': 1, 'sharding_strategy': 'hsdp', 'low_cpu_fsdp': False, 'report_interval': 100, 'checkpoint_interval': 5000, 'use_torch_compile': True, 'data_path': '/proj/data-eng/fsdp/data/R83b', 'datasets': 'CC-MAIN-2023-14,CC-MAIN-2023-40,CC-MAIN-2024-10', 'weights': (18826844689, 16921110027, 13820668539), 'logical_shards': 640, 'learning_rate': 0.0006, 'seq_length': 8192, 'vocab_size': 49152, 'num_steps': 35000, 'fsdp_activation_checkpointing': False, 'batch_size': 2, 'bos_token': None, 'eos_token': 0, 'tracker': 'aim', 'tracker_dir': '/proj/data-eng/fsdp/data/R83b/aim', 'tracker_project_name': '001', 'tracker_run_id': None, 'use_profiler': False}
    # main(kwargs=my_kwargs)

    fire.Fire(main)
