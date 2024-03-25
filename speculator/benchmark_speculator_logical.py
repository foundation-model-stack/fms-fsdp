import argparse
import itertools
import os
import time
import torch
import json

import fms_extras.models.paged_llama
import fms_extras.models.paged_gpt_bigcode
import torch._inductor.config
from fms.models import get_model
from fms.utils import generation, tokenizers
from fms_extras.models.speculator import MLPSpeculator
from fms_extras.utils.generation import paged_generate, speculative_generate
from torch import distributed as dist
from tqdm import tqdm

from fms_fsdp.utils.dataset_utils import Streaming_Doc_Dataset


# This example script measures the logical speedup of running a speculator atop a base model. Run as:
# export CUDA_VISIBLE_DEVICES=1
# e.g., #1: torchrun --nproc_per_node=1 benchmark_speculator_logical.py --architecture=paged_llama --variant=7b --model_path=~/models/7B-F --tokenizer=~/models/tokenizer.model --model_source=hf --speculator_path=~/models/speculator_7B_F.pth --compile --data_path="/path/to/training_dataset_tokens/" --subdata="webhose"
# e.g., #2: torchrun --nproc_per_node=1 benchmark_speculator_logical.py --architecture=paged_gpt_bigcode --variant=ibm.20b --model_path=~/models/granite-20b-instruct --tokenizer=~/models/granite-20b-instruct --model_source=hf --speculator_path=~/models/speculator_granite20B.pth --data_path="/path/to/training_dataset_tokens/" --subdata="github" --n_predict=4 --threshes=[6,4,3,3]

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="paged_llama",
    help="The model architecture to benchmark, e.g. 'paged_llama', 'paged_gpt_bigcode',
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--speculator_path",
    type=str,
    default=None,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--data_path",
    type=str,
    help="Path to the directory containing meta folder with collective file count metadata",
)
parser.add_argument(
    "--subdata",
    type=str,
    help="Subfolder of data_path containing pyarrow shard file(s)",
)
parser.add_argument(
    "--prompt_len",
    type=int,
    default=512,
    help="How many tokens from each document to use as a starter prompt",
)
parser.add_argument(
    "--n_candidates",
    type=int,
    default=5,
    help="How many predictions to evaluate from the speculator at each step",
)

parser.add_argument(
    "--checkpoint_sharding",
    type=str,
    default=None,
    help="type of weight sharding. E.g. tensor-parallel (tp), None",
)

parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--no_flat",
    action="store_true",
    help="Disable batch auto-flattening for handling candidate trees?"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for torch and data loader",
)

parser.add_argument(
    "--n_predict",
    type=int,
    default=3,
    help="Number of speculator heads / number of tokens to guess ahead",
)

parser.add_argument(
    "--threshes",
    type=json.loads,
    default=[6,4,3],
    help="number of top k predictions from each head to generate speculator candidate pool; should be same len as n_predict"
)


args = parser.parse_args()

torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    checkpoint_sharding=args.checkpoint_sharding,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    norm_eps=1e-6,
)
decode_model = None

tokenizer = tokenizers.get_tokenizer(args.tokenizer_path)
model.eval()
torch.set_grad_enabled(False)
speculator = None
if args.speculator_path is not None:
    print("loading speculator")
    speculator = MLPSpeculator(
        model.config.emb_dim, 4096, model.config.src_vocab_size, n_predict=args.n_predict
    )
    speculator.load_state_dict(
        torch.load(args.speculator_path, map_location=device)["model_state"]
    )
    speculator = speculator.to(device)
    print("loading complete on rank", local_rank)

print("initializing paged cache")
# cache setup
from fms_extras.utils.cache.paged import PagedKVCacheManager

use_cache = True
if hasattr(model.config, "kvheads"):
    kv_heads = model.config.kvheads
else:
    kv_heads = 1 if model.config.multiquery_attn else model.config.nheads

kv_cache_manager = PagedKVCacheManager(
    model.config.nlayers,
    model.config.nheads,
    model.config.emb_dim,
    kv_heads=kv_heads,
    tensor_parallel_size=dist.get_world_size() if args.distributed else 1,
    dtype=torch.get_default_dtype(),
    device=device,
    total_num_gpu_blocks=2000,
)
print("cache initialization complete on rank", local_rank)

print("loading dataset", args.data_path)
dataset = Streaming_Doc_Dataset(
    args.data_path,
    local_rank,
    world_size,
    -1,
    datasets=[
        args.subdata,
    ],
    seed=args.seed,
    min_length=2148,
    max_chunksize=8192,
)
dataset = iter(dataset)
data = []
in_middle = False
print("pulling data to build reusable prompt set")
while len(data) < 256:
    chunk = next(dataset)
    if not in_middle:
        data.append(chunk[: args.prompt_len])
    if chunk[-1] == -1:
        in_middle = False
    else:
        in_middle = True
data = torch.IntTensor(data).to(device)

def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def print_result(result, inp, n_steps):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print(f"{len(result) - len(inp)} tokens in {n_steps} steps")
    print()


def infer(ids, k, warmup, model, decode_model, speculator):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    max_seq_len = model.config.max_expected_seq_len if hasattr(model.config, "max_expected_seq_len") else model.config.max_pos

    if k != 0:
        result, n_steps, generated_token_time_out = speculative_generate(
            model,
            ids,
            speculator,
            kv_cache_manager,
            new_tokens=100,
            max_seq_len=max_seq_len,
            decode_model=decode_model,
            top_k=k,
            threshes=args.threshes,
            flatting=not args.no_flat,
        )
    else:
        result, n_steps, generated_token_time_out = paged_generate(
            model,
            ids,
            kv_cache_manager,
            max_new_tokens=100,
            max_seq_len=max_seq_len,
            do_sample=False,
            decode_model=decode_model,
        )
    if not warmup:
        total_tokens = 0
        for i in range(len(result)):
            # print_result(result[i], ids[i], n_steps)
            total_tokens += len(result[i]) - len(ids[i])
        avg_tokens = total_tokens / len(result)
        return generated_token_time_out / avg_tokens, avg_tokens / n_steps
    return None

torch._dynamo.config.cache_size_limit = 64

torch.cuda.empty_cache()
plen = args.prompt_len
bsize = 1
k = args.n_candidates

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    decode_model_ = torch.compile(model, mode=args.compile_mode, fullgraph=True)
    model_ = torch.compile(model, fullgraph=True, dynamic=True)
    speculator_ = torch.compile(speculator, mode=args.compile_mode)
else:
    decode_model_ = model
    model_ = model
    speculator_ = speculator

alltimes = 0
alltokens = 0
ntrials = data.size(0) // bsize
torch.cuda.empty_cache()
for i in tqdm(range(ntrials)):
    inp = data[i * bsize : i * bsize + bsize, :plen]
    if i == 0:
        infer(inp, k, True, model_, decode_model_, speculator_)
    t, tok = infer(inp, k, False, model_, decode_model_, speculator_)
    alltimes += t
    alltokens += tok
print(
    f"prefix = {plen}, bsize = {bsize}, k = {k}, time = {round(alltimes/ntrials, 2)}, tokens/step = {round(alltokens/ntrials, 2)}"
)
print(len(kv_cache_manager.free_blocks))
