### Following parameters are relevant for speculator training: 

- *model_arch*: architecture of the base model (one of: embedllama, embedmixtral, embedgpt_bigcode-- FMS implementations extending the base arch to also emit embedding vector together with the model output. See 'EmbedLLaMA' in train_spculator_utils.py)

- *model_variant*: identifier with which a specific variant (e.g., 7b) is registered for the model architecture. See 'example model registrations' in train_spculator_utils.py. 

- *model_path*: path to dir containing base model weights

- *ckpt_save_path*: path to dir for storing intermediate checkpoints during speculator training

- *ckpt_load_path*: path to dir for loading intermediate speculator checkpoint to resume training

- *sharding_strategy*: how to shard the model across process group: tp / fsdp / hsdp

- *tp_size*: If loading base model using tensor parallel, no. of GPUs/ranks to split the model across 

- *seq_length*: sequence length of the base model

- *batch_size*: batch size for stage 1 training for aligning speculator to base model input behavior

- *report_interval*: no. of steps after which to report training stats

- *checkpoint_interval*: no. of steps after which to save an intermediate speculator checkpoint

- *num_steps*: total no. of speculator training steps (stage 1 + stage 2)

- *stage2_start_step*: no. of steps after which to switch to stage 2 training

- *stage2_batch_size*: batch size for stage 2 training for aligning speculator to base model output behavior

- *n_speculator_heads*: no. of lookahead tokens to train the speculator for

- *speculator_width*: embedding dimension of the speculator MLP

- *use_torch_compile*: whether to compile base model and speculator-- may speed up training.

- *learning_rate*: learning rate for speculator training

- *seed*: random seed to use for training dataset shuffling

- *data_path*: path to dir containing the training dataset. Expects directory to contain subfolders, which in turn contain shard files.

- *datasets*: a list of subdatasets (e.g., commoncrawl, github, etc.) to draw from. If None, draws from all subfolders of data_path.

- *weights*: list of weights reflecting the percentage of tokens to be used from each subdataset during training
