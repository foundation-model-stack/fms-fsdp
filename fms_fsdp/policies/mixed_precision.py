import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy

bfSixteen = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
