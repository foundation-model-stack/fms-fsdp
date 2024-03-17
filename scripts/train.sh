#!/bin/bash

# On AWS, the EFA and OFI paths enable NCCL to use optimized networking.
export LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH

export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

MODEL_ARGS="\
--use_dummy_dataset=False
--ckpt_load_path=/lustre/pretrain/ckpt
--ckpt_save_path=/lustre/pretrain/ckpt
--data_path=/lustre/bluepile-processing/rel0_7/tokens/llama2/high_quality_rerun_fuzzy_deduped
--fsdp_activation_checkpointing=False
--selective_checkpointing=1
--sharding_strategy=hsdp
--low_cpu_fsdp=False
--batch_size=2
--report_interval=200
--checkpoint_interval=20000
--use_torch_compile=False
--use_profiler=False
"

torchrun \
    --nnodes=$SLURM_NTASKS \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=8 \
    --master_addr=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1` \
    --master_port="12234" \
    main_training.py \
    ${MODEL_ARGS}

