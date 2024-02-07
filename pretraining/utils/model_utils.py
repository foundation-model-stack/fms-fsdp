import torch.nn as nn

def post_init_original(model):
    # Override existing fms model init scheme to use the hardcoded one from Llama
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            m.weight.data.normal_(0, 0.02)
    return model
