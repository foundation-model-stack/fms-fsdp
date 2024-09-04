import os
import re
import time
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fms.models import register_model
from fms.models.gpt_bigcode import GPTBigCode
from fms.models.gpt_bigcode import _20b_config as _gpt_bigcode_20b_config
from fms.models.gpt_bigcode import _hf_sd_to_fms_sd as _gptbigcode_hf_sd_to_fms_sd
from fms.models.llama import LLaMA
from fms.models.llama import _hf_sd_to_fms_sd as _llama_hf_sd_to_fms_sd
from fms.models.mixtral import Mixtral, MixtralConfig
from fms.models.mixtral import _hf_sd_to_fms_sd as _mixtral_hf_sd_to_fms_sd
from fms.utils import serialization, tokenizers
from fms.utils.generation import _make_cache_contiguous
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from fms_fsdp.config import train_config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config


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
    include_embeds: bool = True,
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


# Stage 1 training
def stage1_loss(
    cfg, model, speculator, base_model_input, input, loss_fn, ddp_stats, base_model_mesh
):
    """
    Perform a forward pass for stage 1 training and calculate the loss.
    Given the sequence of embeddings produced in parallel by the base model,
    get n+2,n+3,... speculator predictions and compare to ground truth tokens.
    ...
    Args
    ----
    cfg: train_config
        Set of training parameters.
    model: nn.Module
        The frozen base model. Must return output logits AND corresponding embedding vectors.
    speculator: nn.Module
        The speculator to be trained. Takes as input sequence of embeddings and token indices,
        and return token prediction logits for each head.
    input: torch.IntTensor
        The ground truth token indices. If using TP, this is per TP rank,
        with 'base_model_input' containing all-gathered input across all TP ranks
    loss_fn: Callable
        Torch loss function comparing logits to indices i.e. CrossEntropyLoss()
    ddp_stats: torch.FloatTensor
        Aggregate stat tracking buffer.
        Entries are: grad norm, accumulation steps, head 1 loss, head 2 loss, etc.
    base_model_mesh: torch.distributed.device_mesh.DeviceMesh
       Device layout of the particiapting process group ranks
    ----
    Returns: scalar loss value, updated ddp stats, number of tokens in input
    """
    with torch.no_grad():
        _, embeds = model(
            base_model_input[:, : -speculator.n_predict - 1],
            include_embeds=True,
            use_cache=False,
        )
    if cfg.sharding_strategy == "tp":
        embeds = embeds.chunk(base_model_mesh["tp"].size())[
            base_model_mesh["tp"].get_local_rank()
        ]

    preds = speculator(embeds.detach(), input[:, 1:])
    losses = []
    for i in range(preds.size(0)):
        targ = input[:, i + 2 : preds.size(2) + i + 2]  # b n
        loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
        losses.append(loss)
        ddp_stats[2 + i] += loss.item()
    loss = sum(losses)
    return loss, ddp_stats, input.numel()


# Stage 2 training: more heavyweight than stage 1; will take longer
def stage2_loss(
    cfg, model, speculator, base_model_input, input, loss_fn, ddp_stats, base_model_mesh
):
    """
    Perform a forward pass for stage 2 training and calculate the loss.
    Given the sequence of embeddings produced in serial by the base model,
    get n+1,n+2,... speculator predictions and compare to base model's generated tokens.
    Reshapes input to more entries / shorter sequences, for more efficient generation.
    ...
    Args
    ----
    cfg: train_config
        Set of training parameters. Used here for reshaping input batches.
    model: nn.Module
        The frozen base model. Must return output logits AND corresponding embedding vectors.
    speculator: nn.Module
        The speculator to be trained. Takes as input sequence of embeddings and token indices,
        and return token prediction logits for each head.
    input: torch.IntTensor
        The ground truth token indices. If using TP, this is per TP rank,
        with 'base_model_input' containing all-gathered input across all TP ranks
    loss_fn: Callable
        Torch loss function comparing logits to indices i.e. CrossEntropyLoss()
    ddp_stats: torch.FloatTensor
        Aggregate stat tracking buffer.
        Entries are: grad norm, accumulation steps, head 1 loss, head 2 loss, etc.
    base_model_mesh: torch.distributed.device_mesh.DeviceMesh
       Device layout of the particiapting process group ranks
    ----
    Returns: scalar loss value, updated ddp stats, number of tokens in input
    """
    with torch.no_grad():
        grow_factor = cfg.stage2_batch_size // cfg.batch_size
        assert (
            cfg.stage2_prompt_length * grow_factor <= cfg.seq_length
        ), "Error: batch is too small for specified partition"
        base_model_input = base_model_input[
            :, : cfg.stage2_prompt_length * grow_factor
        ].reshape(base_model_input.size(0) * grow_factor, cfg.stage2_prompt_length)
        targs, embeds = generate(
            model,
            base_model_input,
            cfg.seq_length,
            cfg.stage2_seq_length,
            do_sample=True,
            use_cache=True,
            include_embeds=True,
        )

        if cfg.sharding_strategy == "tp":
            targs = targs.chunk(base_model_mesh["tp"].size())[
                base_model_mesh["tp"].get_local_rank()
            ]
            embeds = embeds.chunk(base_model_mesh["tp"].size())[
                base_model_mesh["tp"].get_local_rank()
            ]
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


# on demand checkpointing: echo 1 > /path/to/model_ckpt_dir/do_ckpt
def do_ckpt(ckpt_save_path, reset=False):
    ckpt_cmd_file = ckpt_save_path + "/do_ckpt"
    if not os.path.exists(ckpt_cmd_file):
        return False

    if reset:
        with open(ckpt_cmd_file, "w") as fd:
            fd.write("0")
        return False

    with open(ckpt_cmd_file) as fd:
        if fd.read().strip() == "1":
            return True

    return False


def train_speculator(
    cfg: train_config,
    model: nn.Module,
    speculator: nn.Module,
    local_rank: int,
    rank: int,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpointer: Checkpointer,
    start_step: int = 0,
    n_tok: int = 0,
    profiler: Optional[Union[torch.profiler.profile, None]] = None,
    base_model_mesh=None,
):
    """
    The training loop for speculator training. Handles at a high level: data loading,
    forward and backward passes, model updates, stat tracking, reporting, and checkpointing.
    ...
    Args
    ----
    cfg: train_config
        The set of training parameters
    model: nn.Module
        The frozen base model. Must return output logits AND corresponding embedding vectors.
    speculator: nn.Module
        The speculator to be trained. Takes as input sequence of embeddings and token indices,
        and returns token prediction logits for each head.
    local_rank: int
        The local rank of the current process. Used for stat tracking / aggregation across ranks.
    rank: int
        The global rank of the current process. Used for reporting.
    train_loader: torch.utils.data.DataLoader
        The dataloader used for reading in ground truth token sequences. Train_loader.dataset must
        support save_to_path() for distributed checkpointing via checkpointer.
    optimizer: torch.optim.Optimizer
        The optimizer associated with the speculator's weights
    scheduler: torch.optim.lr_scheduler.LRScheduler
        A scheduler for the optimizer's LR. Scheduler.step() is called on every optimizer step.
    checkpointer: fms_fsdp.utils.checkpointing_utils.Checkpointer
        A checkpointer tied to the save directory. Used for saving distributed checkpoints.
    start_step: optional[int]
        If resuming from checkpoint, resume step count from this value.
    n_tok: optional[int]
        If resuming from checkpoint, resume token count from this value.
    profiler: optional[torch.profiler.profile]
        Optional torch profiler for performance benchmarking.
    base_model_mesh: DeviceMesh
       Device layout of the particiapting process group ranks
    """
    model.eval()
    speculator.train()
    ddp_stats = torch.zeros(2 + speculator.n_predict).to(local_rank)
    train_loss = 0

    start = time.time()
    loop_start = time.time()
    loss_fn = CrossEntropyLoss()
    elapsed_tokens = 0
    for batch_idx, input in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break

        input = input.to(local_rank)

        if cfg.sharding_strategy == "tp":
            base_model_input = torch.zeros(
                base_model_mesh["tp"].size() * input.size(0),
                input.size(1),
                dtype=input.dtype,
                device=input.device,
            )
            dist.all_gather_into_tensor(
                base_model_input, input, group=base_model_mesh["tp"].get_group()
            )
        else:
            base_model_input = input

        optimizer.zero_grad()

        if batch_idx <= cfg.stage2_start_step:
            loss, ddp_stats, step_tok = stage1_loss(
                cfg,
                model,
                speculator,
                base_model_input,
                input,
                loss_fn,
                ddp_stats,
                base_model_mesh,
            )
        else:
            loss, ddp_stats, step_tok = stage2_loss(
                cfg,
                model,
                speculator,
                base_model_input,
                input,
                loss_fn,
                ddp_stats,
                base_model_mesh,
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
                print(f"{time.time()}")
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

        if (
            batch_idx % cfg.checkpoint_interval == 0
            or do_ckpt(cfg.ckpt_save_path) is True
        ):
            torch.cuda.empty_cache()
            checkpointer.save(
                batch_idx,
                speculator,
                optimizer,
                train_loader,
                tokens_seen=elapsed_tokens + n_tok,
            )
            torch.cuda.empty_cache()
            do_ckpt(cfg.ckpt_save_path, reset=True)

    checkpointer.save_single_file(
        batch_idx,
        speculator,
        tokens_seen=elapsed_tokens + n_tok,
        is_compiled=cfg.use_torch_compile,
    )

    return train_loss


class EmbedGPTBigCode(GPTBigCode):
    # Overrides the forward function of GPTBigCode to allow returning embedding vectors
    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
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


class EmbedMixtral(Mixtral):  # FMS impl of Mixtral
    # Overrides the forward function of Mixtral to allow returning embedding vectors
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
        output, cache = self.base_model(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)

        out = [preds]
        if use_cache:
            out.append(cache)
        if include_embeds:
            out.append(output)
        if len(out) == 1:
            return out[0]
        return out


def _gpt_bigcode_factory_factory(config):
    def factory(**kwargs):
        return EmbedGPTBigCode(config, **kwargs)

    return factory


def _llama_factory_factory(config):
    def factory(**kwargs):
        return EmbedLLaMA(config, **kwargs)

    return factory


def _mixtral_factory_factory(config):
    def factory(**kwargs):
        return EmbedMixtral(config, **kwargs)

    return factory


# example model registrations
register_model(
    "embedgpt_bigcode", "20b", _gpt_bigcode_factory_factory(_gpt_bigcode_20b_config)
)
serialization.register_adapter("embedgpt_bigcode", "hf", _gptbigcode_hf_sd_to_fms_sd)

register_model(
    "embedllama", "7b", _llama_factory_factory(get_model_config("llama2_7b"))
)
register_model(
    "embedllama", "8b", _llama_factory_factory(get_model_config("llama3_8b"))
)
serialization.register_adapter("embedllama", "hf", _llama_hf_sd_to_fms_sd)

register_model("embedmixtral", "8x7b", _mixtral_factory_factory(MixtralConfig()))
serialization.register_adapter("embedmixtral", "hf", _mixtral_hf_sd_to_fms_sd)
