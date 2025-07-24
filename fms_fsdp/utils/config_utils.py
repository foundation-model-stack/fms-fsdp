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
    
    print(f"\n== model_variant: {model_variant}")

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
    elif model_variant == "llama2mod_starcoder":
        # llama2 1.4B with starcoder
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2mod_SmolLM135M_like":
        # Vocab: SAME!!!
        # w/o tie_heads: 176,097,600
        # w/ tie_heads: 147,786,048 
        # w/ tie_heads w/kvheads: 134,515,008 (EXACTLY ORIGINAL SmolLM2-135M 134,515,008)!!!
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=576,       # reduced from 2048
            kvheads=3,     
            nheads=9,          
            nlayers=30,        
            max_expected_seq_len=2048, 
            tie_heads = True,
        )
    elif model_variant == "llama2mod_SmolLM135M_like_nokvhead":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=576,
            nheads=9,          
            nlayers=30,        
            max_expected_seq_len=2048, 
            tie_heads = True,
        )
    elif model_variant == "llama2mod_SmolLM135M_like_nokvhead_context8K":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=576,
            nheads=9,          
            nlayers=30,        
            max_expected_seq_len=8192, 
            tie_heads = True,
        )
    elif model_variant == "llama2mod_starcoder135M_context8K_doclingV01":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=640,       # reduced from 2048
            nheads=8,          
            nlayers=15,        
            max_expected_seq_len=8192,  # unchanged
        )
    elif model_variant == "llama2mod_starcoder135M_context2K_doclingV02":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=640,       # reduced from 2048
            nheads=8,          
            nlayers=15,        
            max_expected_seq_len=2048,  # unchanged
        )
    elif model_variant == "llama2mod_starcoder135M_tiehead_doclingV04":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=640,      
            nheads=8,          
            nlayers=15,        
            max_expected_seq_len=8192,  # unchanged
            tie_heads = True,
        )
    elif model_variant == "llama2mod_g4ttk_tieheads_doclingV05":
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=640,
            nheads=8,
            nlayers=15,
            max_expected_seq_len=8192,
            tie_heads = True,
        )
    elif model_variant == "llama2mod_g4ttk_nokvhead": 
        # target 1: as no kvhead means MHA (and no GQA!)
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                
            nheads=9,                   
            nlayers=30,                 
            max_expected_seq_len=8192, 
            tie_heads = True,
        )
    elif model_variant == "llama165m_granite4tiktoken": 
        # diff name for llama2mod_g4ttk_kvhead3 (same name used in fms-fsdp-docling) (can be removed later)
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            kvheads=3,                  # 9/3 = 3 (valid kvhead wrt nheads)
            nlayers=30,                 
            max_expected_seq_len=8192, 
            tie_heads = True,
        )
    elif model_variant == "llama2mod_g4ttk_kvhead3": 
        # target 3: as SAME SmolLM except vocab size but might cause conversion issue:
        # k.view(hf_config.nheads, -1, 2, k.size(1))  (it's unclear why 2?)
        # RuntimeError: shape '[9, -1, 2, 576]' is invalid for input of size 110592 for case of embddim 576
        # https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml
        # TODO: double-check of model conversion!
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=576,                
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            kvheads=3,                  # 9/3 = 3 (valid kvhead wrt nheads)
            nlayers=30,                 
            max_expected_seq_len=8192, 
            tie_heads = True,
        )
    elif model_variant == "llama135m_starcoder": 
        # doctag proj: same arch. with smolLM2-135M
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=576,                # Matches hidden_size
            nheads=9,                   # 576 / 9 = 64 (valid head dim)
            kvheads=3,                  # Matches reference key-value heads
            nlayers=30,                 # set to 22/23 to get 134M/138M params
            max_expected_seq_len=8192, # requested
            hidden_grow_factor=2.6667, # 576 × 2.6667 ≈ 1536 (FFN size)
            multiple_of=16,            # Ensures MLP dim is divisible by 16
            activation_fn="silu",      # Matches hidden_act
            norm_eps=1e-5,             # Matches rms_norm_eps
            attn_bias = False,
            mlp_bias = False,
            tie_heads = True,
        )
    elif model_variant == "llama2mod_starcoder_100352vocab" or model_variant == "llama2mod_granite4tiktoken":
        # llama2 1.4B with starcoder
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2mod_starcoder_3b":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=3072,
            nheads=24,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2mod_starcoder_3b_100352vocab" or model_variant == "llama2mod_granite4tiktoken_3b":
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=3072,
            nheads=24,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2mod_starcoder_7b":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=4096,
            nheads=32,
            nlayers=32,
            hidden_grow_factor=8/3,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2mod_starcoder_7b_100352vocab" or model_variant == "llama2mod_granite4tiktoken_7b":
        llama_config = LLaMAConfig(
            src_vocab_size=100352,
            emb_dim=4096,
            nheads=32,
            nlayers=32,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
        )
    elif model_variant == "llama2_starcoder":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            max_expected_seq_len=8192,
        )
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
    elif model_variant == "llama3mod_1.8b":
        llama_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            max_expected_seq_len=8192,
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
    elif model_variant == "llama3_starcoder":
        llama_config = LLaMAConfig(
            src_vocab_size=49152,
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
    elif model_variant == "llama2_7b":
        #== quicktest/default model
        llama_config = LLaMAConfig()
    else:
        raise ValueError(f"model variant {model_variant} not supported.")

    return llama_config
