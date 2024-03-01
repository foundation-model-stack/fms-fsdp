# FMS FSDP - (Pre)Training FMS with FSDP

The “fms-fsdp” repo is a companion to the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack).
The goal of this repo is to provide a (pre)training example to efficiently train
FMS models, in particular Llama2 by leveraging native PyTorch features - FSDP for training and SDPA implementation of Flash attention v2. While there are many exemplar repositories that can perform pretraining at scale (e.g., [MegatronLM](), [DeepSpeed]()), this work is what IBM has been doing with PyTorch community on using FSDP for training and how to do that efficiently. It is not meant to be an end-to-end framework for training of models, which includes data preparation (pre), and alignment/tuning of the base model (post).

For an end-to-end framework, we would recommend the reader to [OLMo](https://github.com/allenai/OLMo) from AllenAI, which provides datasets, data preprocessing frameworks, leverages FSDP on AMD GPUs for training, and provides a tuning/alignment framework.

## Training throughput benchmarks
We benchmark the best possible throughput and the strategies we employ in the below table and share the throughput obtained on 128 A100 GPUs as well as 96 H100 GPUs, we use the exact same scripts and configurations for these GPUs.

| Model Size | Sharding Strategy | Activation Checkpointing | Batch Size | Training Throughput <br/> A100 80G 128 GPUs <br/> tokens/sec/GPU | Training throughput <br/> H100 96 GPUs <br/> tokens/sec/GPU |
|------------|-------------------|--------------------------|------------|------------------------------------------------------------------|-------------------------------------------------------------|
| 7b         | HSDP              | No AC                    | 2          | 3650                                                             | 7500                                                        |
| 13b        | FSDP              | Selective AC             | 2          | 1800                                                             | 3800                                                        |
| 34b        | FSDP              | Full AC                  | 2          | 700                                                              | 1550                                                        |  
| 70b        | FSDP              | Full AC                  | 2          | 370                                                              | 800                                                         |

We also compute the MFU numbers for each of the above configuration. We use the PyTorch [FLOP counter]() to compute the FLOPS and a theoretical maximum of 312TFLOPS for `bf16` operations on A100 GPUs and 989TFLOPS for H100 GPUs. We cross verify the FLOP counter output with that of math based computation, the latter following the approach outlined in the MegatronLM paper and notice that they are within 2% for the 7B model (384TFLOPS using FLOP counter and 377 using the math formulae. Note that we made changes to the math in the paper to account for lack of activation checkpointing. The MFU numbers are summarized in the below table.

| Model Size | Batch size | MFU (A100 80G) | MFU (H100 80G) |
|------------|------------|----------------|----------------|
| 7B         | 2          | 0.57           | 0.37           |
| 13B        | 2          | 0.59           | 0.40           |
| 34B        | 2          | 0.64           | 0.44           |
| 70B        | 2          | 0.67           | 0.45           |

A few points to note here, on the A100s, we note that for 13B we are not utilizing the hardware as well (only 0.48 MFU) because of smaller batch size. We can dial up the MFU by turning on activation checkpointing, however the throughput falls to 1600 tokens/sec/GPU. Whereas, note that the gaps here are more glaring with H100s where the MFU for 7 and 13B falls below 0.40.

Another point to note here is that for the larger models, we could increase the throughput by a few percentage points when we increase the batch size. However, we have left the batches to be smaller to allow for scaling to 1024 GPUs without introducing tensor parallelism.

## Installation
You need to install the required packages by running the following command.
We recommend running the latest [PyTorch nightlies](https://pytorch.org/) and latest [ibm-fms](https://github.com/foundation-model-stack/foundation-model-stack).
```bash
pip install -r requirements.txt
```

## Training

### Model
We trained one model, a replica of Llama2 7B as an exemplar on IBM curated data. This model was trained to 2.2T tokens with a 4k context length on 128 A100 GPUs for a total of 162k GPU hours, achieving an efficiency of 3700 tokens/sec/GPU (~40B tokens/day), which is roughly 20% faster than the Llama2 published training time. These speedups were possible by combining multiple techniques - SDPA Flash v2 implementation, FSDP with overlap in computation and communication, and selective activation checkpointing.
The generated model has a good performance on various metrics as evaluated by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with MMLU score of 0.5. We share further [scores](docs/evaluation.md) in the details of the model for completeness.

### Dataset
We use an internally curated dataset for training the model. We use sampling ratios similar to what Llama1 paper proposed with minor changes (e.g., no C4 dataset). Since the goal of this repo is to demonstrate the feasibility of training using PyTorch components at scale, we omit the details of the sampling ratios. The overall dataset is roughly 1.5T tokens and the model has seen all the tokens in the dataset at least once.

For this dataset, we designed a large-scale workload dataloader, details can be found [here](docs/dataloader.md).

### Train Config

Below assumes running with Slurm, but same can be easily adopted
if running with other clusters.

1. modify Training Config in [scripts/train.sh](scripts/train.sh) (for the full
list of training configs and best practices, refer to [Configuration Doc](docs/configurations.md)).
2. modify Run Config in [scripts/train.slurm](scripts/train.slurm)

### Run
```bash
sbatch ./scripts/train.slurm
```
For other cluster setup, we can simply use the *torchrun* commands inside `train.sh`.

### Training Details and Lessons learnt
Details on training stability, loss curve, LR curve, etc., as well as what
we have learnt from this journey can be found in [Training Details](docs/train_details.md).

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

## Fine tuning

We have performed preliminary fine-tuning on our base model and details can be found [here](docs/fine_tuning.md). 
