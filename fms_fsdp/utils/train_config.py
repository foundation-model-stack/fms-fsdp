from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "llama3_8b"
    ckpt_load_path: str = "/gpfs/ckpt"
    ckpt_save_path: str = "/gpfs/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/datasets"
    file_type: str = "auto"
    col_name: str = "text,contents,tokens"
    tokenizer_path: str = "/datasets/tokenizers/llama3"
    datasets: str = "cc,wiki"
    weights: str = "0.9,0.1"
    seq_length: int = 8192
    vocab_size: int = 128256
    bos_token: Optional[int] = None
    eos_token: int = 0
    bol_token: Optional[int] = None
    eol_token: Optional[int] = None
    strip_tokens: str = ""
    logical_shards: int = 1024
    num_workers: int = 1

    # fsdp policies
    sharding_strategy: str = "hsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac

    # training spec
    batch_size: int = 2
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2025

    # continued training spec
    resuming_dataset: bool = False

    # profiling
    use_profiler: bool = False

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True
