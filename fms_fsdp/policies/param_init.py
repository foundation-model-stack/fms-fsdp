import torch
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized


# for details, read https://github.com/foundation-model-stack/fms-fsdp/issues/64
def param_init_function(module):
    if (
        isinstance(module, MultiHeadAttention)
        or isinstance(module, WordEmbedding)
        or isinstance(module, GatedLinearUnit)
        or isinstance(module, LayerNormParameterized)
    ):
        module.to_empty(device=torch.cuda.current_device())
        with torch.no_grad():
            module.reset_parameters()
