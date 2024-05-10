import fire
import torch
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def main(model_variant, load_path, save_path, tokenizer_name_or_path):
    print("Initializing model...")
    # 7b
    if model_variant == "mamba_1.5b":
        config_data = {
            "d_model": 2048,
            "n_layer": 48,
            "vocab_size": 128256,
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 8,
        }
    elif model_variant == "mamba_7b":
        config_data = {
            "d_model": 4096,
            "n_layer": 64,
            "vocab_size": 128256,
            "ssm_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 8,
        }
    mamba_config = MambaConfig(**config_data)
    with torch.device("meta"):
        model = MambaLMHeadModel(mamba_config)
    model.to_empty(device="cpu")

    print(f"Reading state dict from {load_path}")
    state_dict = {"model_state": model.state_dict()}
    load_state_dict(
        state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
    )

    print("Loading state dict into the model...")
    model.load_state_dict(state_dict["model_state"])

    print("Saving model to HF-compatible format...")
    model.save_pretrained(save_path)

    print("Copying tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saving at {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
