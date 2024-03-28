import os
import time
from typing import Any, Callable, MutableMapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fms.models.gpt_bigcode import GPTBigCode
from fms.models.llama import LLaMA
from fms.utils.generation import _make_cache_contiguous
from torch.nn import CrossEntropyLoss


def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    include_embeds: bool = False,
):
    """
    A straightforward copy of the generate method in fms.utils.generation.
    The only change is the include_embeds flag, which when true also returns
    the embedding vectors corresponding to the tokens in the output sequence.
    """
    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    embeds = None
    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache
    kwargs["include_embeds"] = include_embeds

    for _ in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]
        output = model(input_ids, **kwargs)
        if not use_cache and not include_embeds:
            logits = output
        else:
            logits = output[0]
            if include_embeds:
                z = output[-1]
            if use_cache:
                past_key_value_states = output[1]
                # TODO: this should go away when reduce-overhead issues are fixed, or
                # maybe could be moved into model code to be more portable.
                if contiguous_cache:
                    kwargs["past_key_value_states"] = _make_cache_contiguous(
                        past_key_value_states
                    )
                else:
                    kwargs["past_key_value_states"] = past_key_value_states
        logits = logits[:, -1, :]

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        result = torch.cat((result, next_val), dim=-1)

        if use_cache:
            next_input = next_val
        else:
            next_input = result

        if include_embeds:
            if embeds is None:
                embeds = z
            else:
                embeds = torch.cat((embeds, z), dim=-2)

    if not batched:
        result = result[0]

    if include_embeds:
        return result, embeds

    return result


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

    checkpointer.save_single_file(
        batch_idx,
        speculator,
        tokens_seen=elapsed_tokens + n_tok,
    )

    return train_loss


class EmbedGPTBigCode(GPTBigCode):
    # Overrides the forward function of GPTBigCode to allow returning embedding vectors
    def forward(
        self,
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
        include_embeds: bool = False,
    ):
        output, cache = self.base_model(
            x,
            mask,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        preds = self.head(output)

        out = [preds]
        if use_cache:
            out.append(cache)
        if include_embeds:
            out.append(output)
        if len(out) == 1:
            return out[0]
        return out


class EmbedLLaMA(LLaMA):
    # Overrides the forward function of LLaMA to allow returning embedding vectors
    def forward(
        self,
        x,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        only_last_token=False,
        attn_algorithm=None,
        include_embeds=False,
    ):
        output, cache = self._helper(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.shared(output, reverse=True)

        out = [preds]
        if use_cache:
            out.append(cache)
        if include_embeds:
            out.append(output)
        if len(out) == 1:
            return out[0]
        return out
