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
        model_config = LLaMAConfig(
            emb_dim=8192,
            multiple_of=4096,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=28672 / 8192,
        )
    elif model_variant == "llama2_34b":
        model_config = LLaMAConfig(
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=48,
            hidden_grow_factor=22016 / 8192,
            max_expected_seq_len=16384,
            rope_theta=1000000.0,
        )
    elif model_variant == "llama2_13b":
        model_config = LLaMAConfig(
            emb_dim=5120,
            nheads=40,
            nlayers=40,
            hidden_grow_factor=13824 / 5120,
        )
    elif model_variant == "llama2_7b":
        model_config = LLaMAConfig(
            hidden_grow_factor=11008 / 4096,
            kvheads=32,
        )
    elif model_variant == "llama2_1.4b":
        model_config = LLaMAConfig(
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            hidden_grow_factor=3,
            kvheads=4,
        )
    elif model_variant == "llama2mod_starcoder135M_context8K_doclingV01":
        model_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=640,       
            nheads=8,          
            nlayers=15,        
            max_expected_seq_len=8192,  
        )
    elif model_variant == "llama135m_starcoder": 
        # doctag proj: same arch. with smolLM2-135M
        model_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            kvheads=3,                  # Matches reference key-value heads
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=2048, 
            hidden_grow_factor=2.6667, # 576 × 2.6667 ≈ 1536 (FFN size)
            multiple_of=16,            # Ensures MLP dim is divisible by 16
            activation_fn="silu",      # Matches hidden_act
            norm_eps=1e-5,             # Matches rms_norm_eps
            attn_bias = False,
            mlp_bias = False,
            tie_heads = True,
        )
    elif model_variant == "llama165m_nokvhead": 
        # name used in prev.exp (with rope 10K): llama165m_granite4tiktoken_nokvhead
        #== same architecture of smolLM2-135M except the embd.layer due to increasing vocab.size for tiktoken
        #== doctag proj: 165M = 134M(smolLM135) - 28M(smolLM's embd.lay) + 57M(embd.layer with vocab 100k)
        model_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=8192, # Matches reference
            hidden_grow_factor=2.6667, # 576 × 2.6667 ≈ 1536 (FFN size)
            rope_theta=100_000.0,
            tie_heads = True,
        )
    elif model_variant == "llama165m": 
        #== name used in prev.exp (with rope 10K): llama165m_granite4tiktoken
        #== same architecture of smolLM2-135M except the embd.layer due to increasing vocab.size for tiktoken
        #== doctag proj: 165M = 134M(smolLM135) - 28M(smolLM's embd.lay) + 57M(embd.layer with vocab 100k)
        model_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            kvheads=3,                  # Matches reference key-value heads
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=8192, # Matches reference
            hidden_grow_factor=2.6667, # 576 × 2.6667 ≈ 1536 (FFN size)
            multiple_of=16,            # Ensures MLP dim is divisible by 16
            activation_fn="silu",      # Matches hidden_act
            norm_eps=1e-5,             # Matches rms_norm_eps
            rope_theta=100_000.0,
            attn_bias = False,
            mlp_bias = False,
            tie_heads = True,
        )
    elif model_variant == "llama165m_granite4tiktoken_wokvhead": 
        model_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=8192, # Matches reference
            rope_theta=100_000.0,
            tie_heads = True,
        )
    elif model_variant == "llama165m_granite4tiktoken_wokvhead_wotieheads": 
        model_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=8192, # Matches reference
        )
    elif model_variant == "llama3_8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_70b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_70b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_194m_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1024,
            nheads=8,
            nlayers=10,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "mamba_9.8b":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    else:
        raise ValueError(f"model variant {model_variant} not supported.")

    return model_config
