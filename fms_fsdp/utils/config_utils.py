from fms.models.llama import LLaMAConfig

from fms_fsdp.config import train_config


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
    if model_variant == "llama2_70b":
        llama_config = LLaMAConfig(
            emb_dim=8192,
            multiple_of=4096,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=28672 / 8192,
        )
    elif model_variant == "llama2_34b":
        llama_config = LLaMAConfig(
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=48,
            hidden_grow_factor=22016 / 8192,
        )
    elif model_variant == "llama2_13b":
        llama_config = LLaMAConfig(
            emb_dim=5120,
            nheads=40,
            nlayers=40,
            hidden_grow_factor=13824 / 5120,
        )
    elif model_variant == "llama2_7b":
        llama_config = LLaMAConfig()
    elif model_variant == "llama2_1.4b":
        llama_config = LLaMAConfig(
            emb_dim=2048,
            nheads=16,
            nlayers=24,
        )
    elif model_variant == "llama3_8b":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama3_8b_4k":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
        )
    elif model_variant == "llama3_1.8b":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama3_1.8b_4k":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
        )
    elif model_variant == "llama3_3.2b":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama3_3.2b_4k":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=4096,
        )
    elif model_variant == "llama3_70b":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama3_70b_4k":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
        )
    else:
        raise ValueError(f"model variant {model_variant} not supported.")

    return llama_config
