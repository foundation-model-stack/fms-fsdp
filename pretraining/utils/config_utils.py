from fms.models.llama import LLaMAConfig

from pretraining.config import train_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def get_model_config(model_variant):
    if model_variant == "70b":
        llama_config = LLaMAConfig(
            src_vocab_size=32000,
            emb_dim=8192,
            norm_eps=1e-05,
            nheads=64,
            nlayers=80,
            hidden_grow_factor=28672 / 8192,
            multiple_of=1,
            activation_fn="silu",
            max_expected_seq_len=2048,
        )
    elif model_variant == "34b":
        llama_config = LLaMAConfig(
            src_vocab_size=32000,
            emb_dim=8192,
            norm_eps=1e-05,
            nheads=64,
            nlayers=48,
            hidden_grow_factor=22016 / 8192,
            multiple_of=1,
            activation_fn="silu",
            max_expected_seq_len=2048,
        )
    elif model_variant == "13b":
        llama_config = LLaMAConfig(
            src_vocab_size=32000,
            emb_dim=5120,
            norm_eps=1e-05,
            nheads=40,
            nlayers=40,
            hidden_grow_factor=13824 / 5120,
            multiple_of=1,
            activation_fn="silu",
            max_expected_seq_len=2048,
        )
    elif model_variant == "7b":
        llama_config = LLaMAConfig(
            src_vocab_size=32000,
            emb_dim=4096,
            norm_eps=1e-05,
            nheads=32,
            nlayers=32,
            hidden_grow_factor=11008 / 4096,
            multiple_of=1,
            activation_fn="silu",
            max_expected_seq_len=2048,
        )
    else:
        raise ValueError(f"model variant {cfg.model_variant} not supported.")

    return llama_config
