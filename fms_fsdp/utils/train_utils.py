import os
from dataclasses import asdict
from functools import partial


try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def batch_losses_from_logits(logits, labels):
    ignore_index = -100

    # uncomment if shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # uncomment if labels are already shifted
    #shift_logits = logits
    #shift_labels = labels

    num_active = (shift_labels != ignore_index).sum(dim=1)
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # reduction='none'

    batch_size, seq_length, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).long()
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss.view(batch_size, -1).sum(dim=1) / num_active

    # pdb.set_trace()
    return loss, num_active

def g_func(losses, l):
    return torch.exp(losses/l)

def h_func(losses, delta=1., l_min=0.75, l_max=1.8):
    return 2. * delta * losses / max(l_max - l_min, 1e-6) - delta * (l_max + l_min) / max(l_max - l_min, 1e-6)

def f_func_mid(losses, delta=1.):
    return 1 - losses**2 / delta**2

def f_func_upper(losses, delta=1.):
    return torch.minimum(losses + delta, delta * torch.ones_like(losses))

def f_func_extremes(losses, delta=1.):
    return torch.abs(losses)

F_FUNC_DIC = {"mid": f_func_mid, "upper": f_func_upper, "extremes": f_func_extremes}

def train(
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
):
    if cfg.tracker:
        if cfg.tracker not in ["wandb", "aim"]:
            raise ValueError(f"tracker {cfg.tracker} not supported.")
        tracker_dir = cfg.tracker_dir
        project_name = cfg.tracker_project_name
        run_id = cfg.tracker_run_id

        if cfg.tracker == "wandb":
            try:
                import wandb  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to wandb but wandb is not installed.")
            if rank == 0:
                print(f"--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(cfg)

        if cfg.tracker == "aim":
            try:
                from aim import Run  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to aim but aim is not installed.")
            if rank == 0:
                print(f"--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(cfg)

    model.train()
    ddp_stats = torch.zeros(3).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    f_func = F_FUNC_DIC[cfg.reweighting_strategy]

    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break

        input = input.to(local_rank)
        label = label.to(local_rank)

        # right here
        optimizer.zero_grad()
        output = model(input)
        # output = output.logits if hasattr(output, "logits") else output
        # ce_loss = torch.nn.CrossEntropyLoss()
        # loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())

        ### todo: our logic goes here ###
        # just sanity check
        # assert hasattr(output, "logits"), "output must have 'logits' attribute"
        # assert hasattr(inputs, "labels"), "input must have 'labels' attribute"

        DEBUG=False
        if rank == 0 and DEBUG:
            print(f'++++++++++++++++++++ logs step: {batch_idx} ++++++++++++++++')
            print(f'inputs: {input}')
            print(f'inputs shape: {input.shape}')
            print(f'logits shape: {output.shape}')
        device_losses, len_norms = batch_losses_from_logits(output, input)

        # Initialize placeholder to hold the losses from all GPUs
        # gathered_losses = [torch.zeros_like(device_losses) for _ in range(dist.get_world_size())]
        gathered_losses = torch.zeros(
            dist.get_world_size(),
            len(device_losses),
            device=device_losses.device,
            dtype=device_losses.dtype
        )
        # All-gather into tensor across all GPUs
        dist.all_gather_into_tensor(gathered_losses, device_losses.detach())

        if batch_idx < cfg.kl_regularizer_warmup_steps:
            kl_regularizer = 100.0
        else:
            kl_regularizer = cfg.kl_regularizer
        with torch.no_grad():
            gathered_losses = gathered_losses.detach()
            perplexity = torch.exp(gathered_losses.mean())

            lmin = gathered_losses.min().item()
            lmax = gathered_losses.max().item()

            h_losses = h_func(gathered_losses.view(-1), delta=1., l_min=lmin, l_max=lmax)
            f_losses = f_func(h_losses, delta=1.)
            g_losses = g_func(f_losses - f_losses.max().item(), l=kl_regularizer)
            weights = g_losses / g_losses.sum()

            device_weights = weights.view(dist.get_world_size(), -1)[local_rank, :]

            if rank == 0 and DEBUG:
                print("gathered_losses.shape",gathered_losses.shape)
                print("gathered_losses",gathered_losses)
                print("device_losses.data",device_losses.data)
                print(f'device weights: {device_weights}')
                print(f'all weights: {weights.view(dist.get_world_size(), -1)}')
                print(f'len norms: {len_norms}')
                print(f'world size: {dist.get_world_size()} and {int(os.environ["WORLD_SIZE"])}')

        # reweight losses and scale up to obtain same scaling as sum of all losses
        loss = torch.sum(device_weights * device_losses) * dist.get_world_size() # * len(device_weights)
        ### end of our logic ####

        loss.backward()
        ddp_stats[1] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        ddp_stats[0] += loss.item()
        ddp_stats[2] += 1

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[2]
            g_norm = ddp_stats[1] / ddp_stats[2]
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            new_tokens_seen = (
                (batch_idx - start_step) * world_size * cfg.batch_size * cfg.seq_length
            )
            if rank == 0:
                total_tokens_seen = tokens_seen + new_tokens_seen
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / (batch_idx - start_step)
                current_throughput = int(
                    cfg.batch_size * cfg.seq_length / current_step_time
                )
                overall_throughput = int(
                    cfg.batch_size * cfg.seq_length / overall_step_time
                )
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("perplexity", perplexity)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print("reserved memory:", reserved_mem)
                print("allocated memory:", allocated_mem)
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

    return train_loss


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_policies(cfg, rank, block):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
    else:
        mixed_precision_policy = None

    # wrapping policy
    wrapping_policy = get_wrapper(block)

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif cfg.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function
    if cfg.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(cfg, rank):
    if not cfg.use_profiler:
        return
    if cfg.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )
