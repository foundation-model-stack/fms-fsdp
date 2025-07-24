#!/bin/bash
# bsub -Is -M 256G -gpu "num=1/task:mode=exclusive_process" /bin/bash  # 1 GPU for better debugging
# cd /proj/data-eng/fsdp/fms-fsdp && conda activate /proj/data-eng/fsdp/train_env
# ./dqa.sh

startTime=$(date +%s) 

SEED=42
MODEL="llama2mod_starcoder"
CHECKPOINT_LOAD_PATH='/proj/data-eng/fsdp/experiments/R83a'
CHECKPOINT_PATH='/proj/data-eng/fsdp/experiments/R83b'

DATA_PATH='/proj/data-eng/fsdp/data/R83b'
# DATASETS="CC-MAIN-2024-10,CC-MAIN-2023-40"
# WEIGHTS="60.0,40.0"
DATASETS="CC-MAIN-2023-14,CC-MAIN-2023-40,CC-MAIN-2024-10"
WEIGHTS="18826844689,16921110027,13820668539"



REPORT_INTERVAL=100
SAVE_STEPS=5000
MAX_STEPS=35000

LOGICAL_SHARDS=640
BOS_TOKEN=None
EOS_TOKEN=0
LEARNING_RATE=6e-4

CODE_PATH="/proj/data-eng/fsdp/fms-fsdp"
ENV_FILE="/proj/data-eng/fsdp/env/train_v01.env"
source ${ENV_FILE}
CONDA_INIT_PATH="/opt/share/miniconda/etc/profile.d/conda.sh" 
source ${CONDA_INIT_PATH}
AIM_LOGS_PATH="/proj/data-eng/fsdp/data/R83b/aim"
JOB_ID=001


VOCAB_SIZE=49152
MAX_SEQ_LEN=8192
GLOBAL_BATCH_SIZE=128 
MAX_BATCH_LEN=2

GPT_ARGS="\
    --seed=${SEED} \
    --model_variant=${MODEL} \
    --use_dummy_dataset=False\
    --ckpt_load_path=${CHECKPOINT_LOAD_PATH}/ \
    --ckpt_save_path=${CHECKPOINT_PATH}/ \
    --selective_checkpointing=1 \
    --sharding_strategy=hsdp \
    --low_cpu_fsdp=False \
    --report_interval=$REPORT_INTERVAL \
    --checkpoint_interval=${SAVE_STEPS} \
    --use_torch_compile=True \
    --data_path="${DATA_PATH}" \
    --datasets="${DATASETS}" \
    --weights="${WEIGHTS}" \
    --logical_shards=${LOGICAL_SHARDS} \
    --learning_rate=${LEARNING_RATE} \
    --seq_length=${MAX_SEQ_LEN} \
    --vocab_size=${VOCAB_SIZE} \
    --num_steps=${MAX_STEPS} \
    --fsdp_activation_checkpointing=False \
    --batch_size=${MAX_BATCH_LEN} \
    --bos_token=${BOS_TOKEN} \
    --eos_token=${EOS_TOKEN} \
    --tracker=aim \
    --tracker_dir=${AIM_LOGS_PATH} \
    --tracker_project_name=${JOB_ID} \
    --tracker_run_id=None \
    --use_profiler=False"



MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=28444 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
JOB_ID=${LSB_JOBID}
NUM_GPUS=$(($NNODES * $GPUS_PER_NODE))


DISTRIBUTED_ARGS="\
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"


#=== Start training
cd $CODE_PATH
# echo -e "\n\n---------------- START TRAINING ..."

# echo -e "\n\n---------------- START TRAINING with $GPT_ARGS...\n"

torchrun $DISTRIBUTED_ARGS main_training_cont.py $GPT_ARGS

endTime=$(date +%s)
elapsed=$(($endTime-$startTime))
echo "== TRAINING \`$EXPERIMENT_NAME\` IS DONE IN: ${elapsed}(s)"
cd $SCRIPT_DIR