import functools

from fms.models.llama import LLaMABlock
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_llama_wrapper():
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LLaMABlock,
        },
    )

    return llama_auto_wrap_policy
