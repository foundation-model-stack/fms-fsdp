# Fine Tuning

To validate that the stack produces a model which is as
easy to fine-tune as Llama models, we convert the FSDP checkpoint into a HF checkpoint and fine tune it
using popular fine-tuning configurations and data mixes.

Specifically, we follow Allen AI’s [open-instruct](https://github.com/allenai/open-instruct) framework, leveraging the TULU v2 stack as-is
(DeepSpeed, TULU v2 mixture and recommended configuration for Llama 2 models). The tuned model
scores are presented below and we note improvements in several tasks. We did not do a hyperparameter
exploration for the best parameters to fine-tune LlamaT. We note that optimal hyperparameter for
LlamaT tuning could be different from Llama 2 as they are likely to have followed different learning
rate schedules.

| Evaluation metric          | Llama2-7B (baseline) | LlamaT-7B |
|----------------------------|----------------------|---------------|
| MMLU (5-shot weighted avg) | 0.53                 | 0.49          |
| Arc challenge              | 0.48                 | 0.43          |
| Arc easy                   | 0.73                 | 0.67          |
| Boolq                      | 0.82                 | 0.82          |
| Copa                       | 0.89                 | 0.86          |
| Hellaswag                  | 0.76                 | 0.75          |
| Openbookqa                 | 0.47                 | 0.42          |
| Piqa                       | 0.79                 | 0.79          |
| Sciq                       | 0.93                 | 0.91          |
| Winogrande                 | 0.71                 | 0.65          |
| Truthfulqa                 | 0.45                 | 0.46          |
| GSM8k (8-shot)             | 0                    | 0             |
