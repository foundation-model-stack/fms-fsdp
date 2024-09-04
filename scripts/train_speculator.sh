#!/bin/bash

# On AWS, the EFA and OFI paths enable NCCL to use optimized networking.
export LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

MODEL_ARGS="\
--model_path=/path/to/models/meta-llama/Llama-2-7b-hf
--model_arch=embedllama
--model_variant=7b
--ckpt_load_path=/path/to/checkpoints/llama2-7b
--ckpt_save_path=/path/to/checkpoints/llama2-7b
--logical_shards=768
--sharding_strategy=hsdp
--seq_length=4096
--batch_size=8
--report_interval=10
--checkpoint_interval=3000
--num_steps=21000
--stage2_start_step=15000
--stage2_batch_size=96
--n_speculator_heads=3
--speculator_width=4096
--use_torch_compile=False
--learning_rate=1e-3
--seed=42
--data_path=/path/to/dataset/
--datasets="'dataset=commoncrawl'"
--weights="'1'"
"

torchrun \
    --nproc_per_node=8 \
    speculator/train_speculator.py \
    ${MODEL_ARGS}


