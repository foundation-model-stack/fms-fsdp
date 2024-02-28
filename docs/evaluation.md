# Preliminary Evaluation

We convert the sharded FSDP checkpoint to a Hugging Face checkpoint and run [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
without any changes. The key scores at the 2T checkpoint are summarized below:

| Evaluation metric          | Llama2-7B (baseline) | LlamaT-7B |
|----------------------------|----------------------|---------------|
| MMLU (zero shot)           | 0.41                 | 0.43          |
| MMLU (5-shot weighted avg) | 0.47                 | 0.50          |
| Arc challenge              | 0.46                 | 0.44          |
| Arc easy                   | 0.74                 | 0.71          |
| Boolq                      | 0.78                 | 0.76          |
| Copa                       | 0.87                 | 0.83          |
| Hellaswag                  | 0.76                 | 0.74          |
| Openbookqa                 | 0.44                 | 0.42          |
| Piqa                       | 0.79                 | 0.79          |
| Sciq                       | 0.91                 | 0.91          |
| Winogrande                 | 0.69                 | 0.67          |
| Truthfulqa                 | 0.39                 | 0.39          |
| GSM8k (8-shot)             | 0.13                 | 0.11          |


