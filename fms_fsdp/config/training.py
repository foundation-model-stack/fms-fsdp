from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/fsx/data"
    file_type: str = "arrow"
    col_name: str = "tokens"
    tokenizer_path: str = "/fsx/tokenizer"
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange"
    weights: str = "7725,500,550,28,17,22,25,8,100,500,175,250,100"
    seq_length: int = 4096
    vocab_size: int = 32000
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
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # continued training spec
    resuming_dataset: bool = False

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True

    # speculator training
    tp_size: int = 8
    model_arch: str = "embedllama"
    model_path: str = "/path/to/model/"
    n_speculator_heads: int = 3
    speculator_width: int = 4096
    speculator_tie_weights: bool = True
    speculator_scale_input: bool = True
    stage2_start_step: int = 15000
    stage2_prompt_length: int = 64
    stage2_batch_size: int = 96
    stage2_seq_length: int = 256

    # FIM training
    fim_training: bool = False
    psm_rate: float = 0.0
    spm_rate: float = 0.0
    fim_pre: int = 1
    fim_mid: int = 2
    fim_suf: int = 3
