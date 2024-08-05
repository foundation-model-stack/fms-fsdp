import fire
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict

from fms_fsdp.utils.config_utils import get_model_config


def main(model_variant, load_path, save_path, tokenizer_name_or_path):
    print("Initializing model...")
    config_data = get_model_config(model_variant)
    mamba_config = MambaConfig(**config_data)
    model = MambaLMHeadModel(mamba_config)

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
