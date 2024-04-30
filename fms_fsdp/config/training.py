from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/lustre/pretrain/ckpt"
    ckpt_save_path: str = "/lustre/pretrain/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/lustre/data"
    seq_length: int = 8192
    sep_token: int = 1
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange,lang=en/dataset=PG19"
    weights: str = "7700,500,550,28,17,22,25,8,100,500,175,250,100,25"
    logical_shards: int = 800

    # fsdp policies
    mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    sharding_strategy: str = "hsdp"
    low_cpu_fsdp: bool = False

    # training spec
    seed: int = 2023
    batch_size: int = 2
    num_steps: int = 2000000
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 200
    checkpoint_interval: int = 20000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/lustre/lchu/fms-fsdp"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = False

    # speculator training
    model_path: str = "/lustre/llama_weights/8B-llama3-hf"
    n_speculator_heads: int = 3
    speculator_width: int = 4096
    stage2_start_step: int = 15000
    stage2_prompt_length: int = 64
    stage2_batch_size: int = 12
    stage2_seq_length: int = 256
