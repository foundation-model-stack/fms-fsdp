# FMS FSDP - (Pre)Training FMS with FSDP

The "fms-fsdp" repo is a companion to the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack).
The goal of this repo is to provide a (pre)training example to efficiently train
FMS models (specifically, our FMS implementation of Llama) by leveraging PyTorch
FSDP.

## Installation
You need to install the required packages by running the following command.  
We recommend running the latest PyTorch nightly and latest ibm-fms.

```bash
pip install -r requirements.txt
```

## Training

### Dataset
TODO

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



