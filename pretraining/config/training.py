from dataclasses import dataclass


@dataclass
class train_config:
    # seed
    seed: int = 2023

    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/lustre/pretrain/ckpt"
    ckpt_save_path: str = "/lustre/pretrain/ckpt"

    # data and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/lustre/data"
    seq_length: int = 4096
    sep_token: int = 1
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange,lang=en/dataset=PG19"
    weights: str = "7700,500,550,28,17,22,25,8,100,500,175,250,100,25"
    logical_shards: int = 768

    # compile
    use_torch_compile: bool = False

    # profiler
    use_profiler: bool = False

    # fsdp policies
    mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: int = 1
    sharding_strategy: str = "hsdp"
    sharding_group_size: int = 8
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_steps: int = 2000000
    learning_rate: float = 3e-4

    # reporting
    report_interval: int = 200
    checkpoint_interval: int = 20000
