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


def apply_fsdp_checkpointing(model, every_xth_item):
    def selective_checkpointing(submodule):
        selective_checkpointing.__dict__.setdefault("_count", 0)

        if isinstance(submodule, LLaMABlock):
            selective_checkpointing._count += 1
            if (
                not every_xth_item
                or selective_checkpointing._count % every_xth_item == 0
            ):
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )
