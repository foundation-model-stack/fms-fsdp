import os
import time

import torch
import torch.distributed as dist
from fms.utils.generation import generate
from torch.nn import CrossEntropyLoss


def stage1_loss(model, speculator, input, loss_fn, ddp_stats):
    with torch.no_grad():
        _, embeds = model(
            input[:, : -speculator.n_predict - 1],
            include_embeds=True,
            use_cache=False,
        )
    preds = speculator(embeds.detach(), input[:, 1:])

    losses = []
    for i in range(preds.size(0)):
        targ = input[:, i + 2 : preds.size(2) + i + 2]  # b n
        loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
        losses.append(loss)
        ddp_stats[2 + i] += loss.item()
    loss = sum(losses)
    return loss, ddp_stats, input.numel()


def stage2_loss(cfg, model, speculator, input, loss_fn, ddp_stats):
    with torch.no_grad():
        grow_factor = cfg.stage2_batch_size // cfg.batch_size
        assert (
            cfg.stage2_prompt_length * grow_factor <= cfg.seq_length
        ), "Error: batch is too small for specified partition"
        input = input[:, : cfg.stage2_prompt_length * grow_factor].reshape(
            input.size(0) * grow_factor, cfg.stage2_prompt_length
        )
        targs, embeds = generate(
            model,
            input,
            cfg.seq_length,
            cfg.stage2_seq_length,
            do_sample=True,
            use_cache=True,
            include_embeds=True,
        )
        targs = targs[:, -cfg.stage2_seq_length :]
        embeds = embeds[:, -cfg.stage2_seq_length : -speculator.n_predict]
    preds = speculator(embeds.detach(), targs[:, :-1].detach())

    losses = []
    for i in range(preds.size(0)):
        targ = targs[:, i + 1 : preds.size(2) + i + 1]  # b n
        loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
        losses.append(loss)
        ddp_stats[2 + i] += loss.item()
    loss = sum(losses)
    return loss, ddp_stats, targs.numel()


def train_speculator(
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
    n_tok,
):
    model.eval()
    speculator.train()
    ddp_stats = torch.zeros(2 + speculator.n_predict).to(local_rank)

    start = time.time()
    loop_start = time.time()
    loss_fn = CrossEntropyLoss()
    elapsed_tokens = 0
    for batch_idx, input in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        input = input.to(local_rank)

        optimizer.zero_grad()

        if batch_idx <= cfg.stage2_start_step:
            loss, ddp_stats, step_tok = stage1_loss(
                model, speculator, input, loss_fn, ddp_stats
            )
        else:
            loss, ddp_stats, step_tok = stage2_loss(
                cfg, model, speculator, input, loss_fn, ddp_stats
            )

        loss.backward()
        ddp_stats[0] += speculator.clip_grad_norm_(cfg.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        ddp_stats[1] += 1

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[2:] / ddp_stats[1]
            g_norm = ddp_stats[0] / ddp_stats[1]
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            elapsed_tokens += cfg.report_interval * world_size * step_tok
            if rank == 0:
                print("step:", batch_idx)
                print("tokens seen:", n_tok + elapsed_tokens)
                for i in range(len(train_loss)):
                    print(f"loss {i+1}:", train_loss[i].item())
                print("gradient norm:", g_norm.item())
                print(
                    f"speed for these {cfg.report_interval} steps:",
                    (time.time() - start) / cfg.report_interval,
                )
                print("overall speed:", elapsed_time / (batch_idx - start_step))
                print("LR:", scheduler.get_last_lr())
                print(
                    "reserved memory:",
                    torch.cuda.max_memory_reserved(device=torch.cuda.current_device()),
                )
                print(
                    "active memory:",
                    torch.cuda.max_memory_allocated(device=torch.cuda.current_device()),
                )
                print(
                    "overall token per gpu per sec:",
                    int(elapsed_tokens / world_size / elapsed_time),
                )
                print("token per day:", int(elapsed_tokens / elapsed_time * 3600 * 24))
                print()
            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            torch.cuda.empty_cache()
            checkpointer.save(
                batch_idx,
                speculator,
                optimizer,
                train_loader,
                tokens_seen=elapsed_tokens + n_tok,
            )
            torch.cuda.empty_cache()

    return train_loss
