from functools import partial

from fms.models.llama import LLaMABlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, selectivity):
    block_idx = 0
    m, n = selectivity

    def selective_checkpointing(submodule):
        nonlocal block_idx

        if isinstance(submodule, LLaMABlock):
            current_block_idx = block_idx
            block_idx += 1
            if current_block_idx % n in range(m):
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )
