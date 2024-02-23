import torch
from torch.nn import CrossEntropyLoss
import time
import os
import torch.distributed as dist


def stage1_loss(inp, out, preds, seqlen, ddp_stats):
    losses = []
    loss_fn = CrossEntropyLoss()
    for i in range(preds.size(0)):
        targ = inp[:, i + 2 : seqlen + i + 2]  # b n
        loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
        losses.append(loss)
        ddp_stats[2+i] += loss.item()
    return sum(losses), ddp_stats

def train_speculator_stage1(
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
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        with torch.no_grad():
            _, embeds = model(
                input[:, :-cfg.n_speculator_heads - 1],
                include_embeds=True,
                use_cache=False,
            )
        preds = speculator(embeds.detach(), input[1:])

        losses = []
        loss_fn = CrossEntropyLoss()
        for i in range(preds.size(0)):
            targ = input[:, i + 2 : cfg.seq_length + i + 2]  # b n
            loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
            losses.append(loss)
            ddp_stats[2+i] += loss.item()
        loss = sum(losses)
        
        loss.backward()
        ddp_stats[0] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
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
            elapsed_tokens = (
                (batch_idx - start_step) * world_size * cfg.batch_size * cfg.seq_length
            )
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
            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                speculator,
                optimizer,
                train_loader,
                tokens_seen=elapsed_tokens + n_tok,
            )

    return train_loss