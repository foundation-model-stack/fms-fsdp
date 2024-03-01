# Configurations

All configurations in [scripts/train.sh](scripts/train.sh) will be passed into
[training configs](../pretraining/config/training.py). 

## Full list of configurations
     
### Model
  - **model_variant**: the llama variant, values in "7b", "13b", "34b" and "70b".
  - **ckpt_load_path**: the path from where checkpoint will be loaded for continued training.
  - **ckpt_save_path**: the path to which checkpoint will be saved.
    
### Dataset and Dataloader 
  - **use_dummy_dataset**: set this to `True`` to use dummy dataset for quick testing and performance benchmarking.
  - **data_path**: Data path.
  - **seq_length**: Sequence/context length to build when preparing model input.
  - **sep_token**: Separator token in the tokenized dataset.
  - **datasets**: Subfolders under `datapath` that contains different datasets. 
  - **weights**: Proportion of each dataset when training.
  - **logical_shards**: Number of logical shards when building dataloader. This is an advanced setting and we will go into detail in a future update.
    
### FSDP policies
  - **sharding_strategy**: "FSDP" (Fully Sharded) or "HSDP" (Hybrid Sharded), HSDP allows for sharding within a node and DP across nodes.
  - **mixed_precision**: Whether to use `bf16` mixed precision for training 
  - **fsdp_activation_checkpointing**: whether to turn on activation checkpointing 
  - **selective_checkpointing**: How many blocks to checkpoint the activation. 1 is the default setting, experiment with this number to trade off between memory and compute requirements.
  - **low_cpu_fsdp**: Whether to load the model in low cpu mode. This is useful when loading large models like 70b.

### Training spec
  - **seed**: random seed for reproduction. 
  - **batch_size**: batch size per gpu. 
  - **num_steps**: total number of steps to train. 
  - **learning_rate**: learning rate. This is the max learning rate for model to warm up to. We default to the `cosine` schedule which is popular for pretraining workloads.
    
### Profiling and reporting 
  - **use_profiler**: whether to turn on PyTorch profiler to generate profiling traces
  - **report_interval**: how many steps to report training metrics
  - **checkpoint_interval**: how many steps to save a checkpoint

### Compile 
  - **use_torch_compile**: whether to turn on compile. It is recommended to NOT use compile at this stage due to some known issues with compile-training.


## Deep Dive into FSDP Configs
You can skip this section if you are already familiar with FSDP.

### Basics
In case you are new to FSDP, here are some basic references:  
[FSDP Intro](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)  
[FSDP API](https://pytorch.org/docs/stable/fsdp.html)

### Tune the configs
The key to achieve the best performance for FSDP is to fully overlap the communication
time with computation time so that the GPUs are busy all the time as if there is no
communication cost/gap.

**sharding_strategy** is the first thing you want to decide. FSDP will shard your model
across all devices (GPUs) while HSDP will only shard your model across devices within
the same node. E.g. if you are training with 128 nodes (i.e. 128 * 8 = 1024 total gpus),
FSDP will shard your model across all 1024 gpus while HSDP will shard your model across
8 gpus in each node. Therefore, FSDP will save more memory while HSDP will introduce
lesser communications.  For smaller models like 7b and 13b, HSDP is preferred as it
will make communication shorter thus easier to be overlapped with computation; while
for larger models like 34b and 70b, FSDP will be a necessity as the model is too large
to be fitted into only 8 gpus.

**fsdp_activation_checkpointing** controls if activation checkpointing is enabled. 
Enabling it will greatly save the memory but also increase the computation time due
to activation re-computation. For large models you would typically have to set this
to `True` as activations consume large amount of memory and without checkpointing it
you will face OOM. For smaller models, depending on your batch size, you may or may
not enable it. As a companion, **selective_checkpointing** controls how "often"
to checkpoint the activation (i.e. checkpoint activation only every k steps), the 
smaller the value, the more often it checkpoints and thus the more memory will
be saved. default value is 1 meaning checkpoint every block.
