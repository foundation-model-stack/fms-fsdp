# FMS FSDP - (Pre)Training FMS with FSDP

The “fms-fsdp” repo is a companion to the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack).
The goal of this repo is to provide a (pre)training example to efficiently train
FMS models, in particular Llama2 by leveraging native PyTorch features - FSDP for training and SDPA implementation of Flash attention v2. While there are many exemplar repositories that can perform pretraining at scale (e.g., [MegatronLM](), [DeepSpeed]()), this work is what IBM has been doing with PyTorch community on using FSDP for training and how to do that efficiently. It is not meant to be an end-to-end framework for training of models, which includes data preparation (pre), and alignment/tuning of the base model (post).

For an end-to-end framework, we would recommend the reader to [OLMo](https://github.com/allenai/OLMo) from AllenAI, which provides datasets, data preprocessing frameworks, leverages FSDP on AMD GPUs for training, and provides a tuning/alignment framework.

| Model Size | Hardware    | Sharding Strategy | Activation Checkpointing | Batch Size | Training Throughput <br/> (128 GPUs) | Example Script        | Profile Trace                                                         | 
|------------|-------------|-------------------|--------------------------|------------|--------------------------------------|-----------------------|-----------------------------------------------------------------------|
| 7b         | 128 * A100  | HSDP              | False                    | 2          | **3760 token/gpu/sec**               | [7b](scripts/7b.sh)   | [7b trace](https://ibm.box.com/s/ohaliqku0rl52jc9dhw1cb04opgssgy3)    |
| 13b        | 128 * A100  | HSDP              | False                    | 1          | **1700 token/gpu/sec**               | [13b](scripts/13b.sh) | [13b trace](https://ibm.box.com/s/2j0uib7m1p5wqjhv9dagq4331n62iyv6)   |
| 34b        | 128 * A100  | FSDP              | True                     | 8          | **772 token/gpu/sec**                | [34b](scripts/34b.sh) | [34b trace](https://ibm.box.com/s/tf7x6254egzgzrn6ceh6kgdgy0rbowz6)   |  
| 70b        | 128 * A100  | FSDP              | True                     | 6          | **380 token/gpu/sec**                | [70b](scripts/70b.sh) | [70b trace](https://ibm.box.com/s/5o1ohr1144nloqjrelsvunrq0rutyynu)   |


## Installation
You need to install the required packages by running the following command.
We recommend running the latest [PyTorch nightlies](https://pytorch.org/) and latest [ibm-fms](https://github.com/foundation-model-stack/foundation-model-stack).
```bash
pip install -r requirements.txt
```

## Train

### Model
We trained one model, a replica of Llama2 7B as an exemplar on IBM curated data, the [Blue Pile](https://mitibmwatsonailab.mit.edu/research/blog/creating-space-for-the-evolution-of-generative-and-trustworthy-ai/). This model was trained to 2.2T tokens with a 4k context length on 128 A100 GPUs for a total of 162k GPU hours, achieving an efficiency of 3500 tokens/sec/GPU (~38B tokens/day), which is roughly 20% faster than the Llama2 published training time. These speedups were possible by combining multiple techniques - SDPA Flash v2 implementation, FSDP with overlap in computation and communication, and selective activation checkpointing.
The generated model has a good performance on various metrics as evaluated by [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness), with MMLU score of 0.5. We share further [scores](docs/evaluation.md) in the details of the model for completeness.

### Dataset
We use an internally curated dataset, Blue Pile for training the model. We use sampling ratios similar to what Llama1 paper proposed with minor changes (e.g., no C4 dataset). Since the goal of this repo is to demonstrate the feasibility of training using PyTorch components at scale, we omit the details of the sampling ratios. The overall dataset is roughly 1.5T tokens and the model has seen all the tokens in the dataset at least once.

For this dataset, we designed a large-scale workload dataloader, details can be found [here](docs/dataloader.md).

### Configuration

Below assumes running with Slurm, but same can be easily adopted
if running with OCP.

1. modify Training Config in [scripts/train.sh](scripts/train.sh) (for the full
list of training configs and best practices, refer to [Configuration Doc](docs/configurations.md)).
2. modify Run Config in [scripts/train.slurm](scripts/train.slurm)

### Run
```bash
sbatch ./scripts/train.slurm
```
For other cluster setup, we can simply use the *torchrun* commands inside `train.sh`.

## Post Training

### Convert to Hugging Face format

The model trained with this repo is in FMS format, and you might want to convert it
to Huggingface format so that you can load it natively with Huggingface and leverage Huggingface ecosystem:
```bash
python fms_to_hf.py --model_variant 7b --load_path /path/to/trained/checkpoints --save_path /output/path --tokenizer_name_or_path /path/to/llama/tokenizer
```
> [!Note]
> This repo consumes pre-tokenized data thus does not require a tokenizer. However,
> Huggingface checkpoint requires a paired tokenizer thus you need to pass a tokenizer
> here so it can be copied over to the save dir. Just download the HF Llama tokenizer
> and pass the path here.



