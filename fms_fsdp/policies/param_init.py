import torch
from fms.modules.attention import MultiHeadAttention, QKV
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized


# for details, read https://github.com/foundation-model-stack/fms-fsdp/issues/64
def param_init_function(module, cfg):
    scales = {
        MultiHeadAttention: cfg.mup_attn_init,
        QKV: cfg.mup_attn_init,
        GatedLinearUnit: cfg.mup_ffn_init,
        WordEmbedding: 1,
        LayerNormParameterized: 1,
    }
    scale_keys = list(scales.keys())
    scale_vals = list(scales.values())
    type_id = [isinstance(module, x) for x in scale_keys]
    is_resettable = sum(type_id)
    if is_resettable:
        module_type_id = type_id.index(True)
        module.to_empty(device=torch.cuda.current_device())
        with torch.no_grad():
            module.reset_parameters(scale=scale_vals[module_type_id])
