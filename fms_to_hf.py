import fire
import torch
from fms.models.hf import to_hf_api
from fms.models.llama import LLaMA
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict
from transformers import LlamaConfig, LlamaForCausalLM

from fms_fsdp.utils.config_utils import get_model_config


def convert_to_hf(model: LLaMA, model_variant, is_old_fms) -> LlamaForCausalLM:
    fms_hf_model = to_hf_api(model)
    hf_config = fms_hf_model.config
    if "llama3" in model_variant:
        hf_config.bos_token_id = 128000
        hf_config.eos_token_id = 128001
    oss_hf_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            rms_norm_eps=hf_config.norm_eps,
            num_attention_heads=hf_config.nheads,
            num_key_value_heads=None if hf_config.kvheads == 0 else hf_config.kvheads,
            num_hidden_layers=hf_config.nlayers,
            intermediate_size=hf_config.multiple_of
            * (
                (
                    int(hf_config.hidden_grow_factor * hf_config.hidden_size)
                    + hf_config.multiple_of
                    - 1
                )
                // hf_config.multiple_of
            ),
            pad_token_id=(
                None if hf_config.pad_token_id == -1 else hf_config.pad_token_id
            ),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            max_position_embeddings=hf_config.max_expected_seq_len,
        )
    )

    # compute the freq from rot_emb since it is gathered lazily
    rot_emb = fms_hf_model.decoder.model.rot_emb
    max_seq_len = rot_emb.max_seq_len
    alpha = rot_emb._alpha(max_seq_len)
    ratio = rot_emb.ratio
    dim = rot_emb.dim
    if rot_emb.ntk_scaling:
        ratio = ratio * alpha ** (dim / (dim - 2))
    freqs = 1.0 / (ratio ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    with torch.no_grad():
        oss_hf_model.model.embed_tokens.weight.copy_(fms_hf_model.embedding.weight)
        i = 0
        for oss_hf_layer in oss_hf_model.model.layers:
            fms_hf_layer = fms_hf_model.decoder.model.layers[i]

            # self attn
            if is_old_fms:
                oss_hf_layer.self_attn.q_proj.weight.copy_(
                    fms_hf_layer.attn.query.weight
                )
                oss_hf_layer.self_attn.k_proj.weight.copy_(fms_hf_layer.attn.key.weight)
                oss_hf_layer.self_attn.v_proj.weight.copy_(
                    fms_hf_layer.attn.value.weight
                )
            else:
                q, k, v = torch.split(
                    fms_hf_layer.attn.in_proj.qkv_fused.weight,
                    fms_hf_layer.attn.in_proj.splits,
                    dim=0,
                )
                oss_hf_layer.self_attn.q_proj.weight.copy_(q)
                oss_hf_layer.self_attn.k_proj.weight.copy_(k)
                oss_hf_layer.self_attn.v_proj.weight.copy_(v)
            oss_hf_layer.self_attn.o_proj.weight.copy_(fms_hf_layer.attn.dense.weight)
            oss_hf_layer.self_attn.rotary_emb.inv_freqs = freqs

            # mlp
            if is_old_fms:
                oss_hf_layer.mlp.gate_proj.weight.copy_(
                    fms_hf_layer.ff_sub_layer.wg.weight
                )
                oss_hf_layer.mlp.up_proj.weight.copy_(
                    fms_hf_layer.ff_sub_layer.w1.weight
                )
            else:
                wg1_fused = fms_hf_layer.ff_sub_layer.wg1_fused.weight
                wg_splits = [wg1_fused.size(0) // 2, wg1_fused.size(0) // 2]
                wg, w1 = torch.split(
                    fms_hf_layer.ff_sub_layer.wg1_fused.weight, wg_splits, dim=0
                )
                oss_hf_layer.mlp.gate_proj.weight.copy_(wg)
                oss_hf_layer.mlp.up_proj.weight.copy_(w1)
            oss_hf_layer.mlp.down_proj.weight.copy_(fms_hf_layer.ff_sub_layer.w2.weight)

            # layer norm
            oss_hf_layer.input_layernorm.weight.copy_(fms_hf_layer.ln.weight)
            oss_hf_layer.post_attention_layernorm.weight.copy_(
                fms_hf_layer.ff_ln.weight
            )

            # adjust q, k
            q = oss_hf_layer.self_attn.q_proj.weight.data
            q = (
                q.view(hf_config.nheads, -1, 2, q.size(1))
                .transpose(1, 2)
                .reshape(*q.size())
            )
            oss_hf_layer.self_attn.q_proj.weight.copy_(q)

            k = oss_hf_layer.self_attn.k_proj.weight.data
            k = (
                k.view(
                    hf_config.nheads if hf_config.kvheads == 0 else hf_config.kvheads,
                    -1,
                    2,
                    k.size(1),
                )
                .transpose(1, 2)
                .reshape(*k.size())
            )
            oss_hf_layer.self_attn.k_proj.weight.copy_(k)

            i = i + 1
        oss_hf_model.model.norm.weight = fms_hf_model.decoder.model.dec_norm.weight
        oss_hf_model.lm_head.weight = fms_hf_model.lm_head.weight

    return oss_hf_model


def main(
    model_variant, compiled, is_old_fms, load_path, save_path, tokenizer_name_or_path
):
    print("Initializing model...")
    llama_config = get_model_config(model_variant)
    with torch.device("meta"):
        model = LLaMA(llama_config)
    model.to_empty(device="cpu")

    print(f"Reading state dict from {load_path}")
    if not compiled:
        state_dict = {"model_state": model.state_dict()}
    else:
        state_dict = {"model_state": {"_orig_mod": model.state_dict()}}
    load_state_dict(
        state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
    )

    print("Loading state dict into the model...")
    if not compiled:
        model.load_state_dict(state_dict["model_state"])
    else:
        model.load_state_dict(state_dict["model_state"]["_orig_mod"])

    print("Converting to HF model..")
    hf_model = convert_to_hf(model, model_variant, is_old_fms)
    hf_model.save_pretrained(save_path)

    print("Copying tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model converted to HF model, saving at {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
