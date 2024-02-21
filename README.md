# FMS FSDP - (Pre)Training FMS with FSDP

The "fms-fsdp" repo is a companion to the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack).
The goal of this repo is to provide a (pre)training example which could efficiently train
FMS models (specifically, our FMS implementation of Llama) by leveraging PyTorch
FSDP.

| Model Size | Hardware    | Sharding Strategy | Activation Checkpointing | Batch Size | Training Throughput <br/> (128 GPUs) | Example Script        | Profile Trace                                                         | 
|------------|-------------|-------------------|--------------------------|------------|--------------------------------------|-----------------------|-----------------------------------------------------------------------|
| 7b         | 128 * A100  | HSDP              | False                    | 2          | 3760 token/gpu/sec                   | [7b](scripts/7b.sh)   | [7b trace](https://ibm.box.com/s/ohaliqku0rl52jc9dhw1cb04opgssgy3)    |
| 13b        | 128 * A100  | HSDP              | False                    | 1          | 1700 token/gpu/sec                   | [13b](scripts/13b.sh) | [13b trace](https://ibm.box.com/s/2j0uib7m1p5wqjhv9dagq4331n62iyv6)   |
| 34b        | 128 * A100  | FSDP              | True                     | 8          | 772 token/gpu/sec                    | [34b](scripts/34b.sh) | [34b trace](https://ibm.box.com/s/tf7x6254egzgzrn6ceh6kgdgy0rbowz6)   |  
| 70b        | 128 * A100  | FSDP              | True                     | 6          | 380 token/gpu/sec                    | [70b](scripts/70b.sh) | [70b trace](https://ibm.box.com/s/5o1ohr1144nloqjrelsvunrq0rutyynu)   |


## Installation
You need to install the required packages by running the following command.  
We recommend running the latest PyTorch nightly and latest ibm-fms.

```bash
pip install -r requirements.txt
```

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



