import torch
from torch.utils.data.distributed import DistributedSampler

from pretraining.utils.dataset_utils import (
    Scalable_Sampling_Dataset,
    Streaming_Doc_Dataset,
    Buffer_Dataset,
    Preload_Buffer_Dataset,
    Preprocess_Dataset,
)


def get_dummy_loader(cfg, rank, world_size):
    class SteadyCounter(torch.utils.data.IterableDataset):
        # Spit out incremental counts of constant length l, modulo vocab size v
        def __init__(self, l, v):
            self.i = 0
            self.rank = 0
            self.worldsize = 1
            self.l = l
            self.v = v

        def __iter__(self):
            while True:
                out = torch.IntTensor(
                    [x % self.v for x in range(self.i, self.i + self.l)]
                )
                yield out, out
                self.i += self.l

    data = SteadyCounter(
        cfg.seq_length, 10000
    )  # 10k vsize since vsize isn't actually in the cfg
    return torch.utils.data.DataLoader(data, batch_size=cfg.batch_size)


def get_data_loader(cfg, rank, world_size):
    datasets, weights = parse_data_args(cfg.datasets, cfg.weights)

    def causal_lm(data_seq, prompt_len=1):
        """
        Perform causal language modeling by right-shifting the input sequence.
        Sets first prompt_len tokens to be ignored by the loss. Assumes inputs start with BOS.
        """
        data_seq = torch.IntTensor(data_seq)
        t = data_seq.clone()[1:]
        data_seq = data_seq[:-1]
        t[:prompt_len] = -100
        return data_seq, t

    base_scalable = Scalable_Sampling_Dataset
    data = base_scalable(
        cfg.data_path,
        Streaming_Doc_Dataset,
        rank,
        world_size,
        cfg.sep_token,
        trainsplit=1,
        is_val=False,
        min_length=3,
        datasets=datasets,
        weights=weights,
        seed=cfg.seed,
        verbose=(rank == 0),
        n_logical_shards=cfg.logical_shards,
    )
    data = Buffer_Dataset(
        data,
        [cfg.seq_length + 1],
        drop_final_token=cfg.sep_token,
        pack_hard=True,
    )
    data = Preload_Buffer_Dataset(data, 10000)
    data = Preprocess_Dataset(data, causal_lm)

    return torch.utils.data.DataLoader(data, num_workers=0, batch_size=cfg.batch_size)


def parse_data_args(datas, weights):
    def splitstrip(x):
        return [item.strip() for item in x.split(",")]

    datas = splitstrip(datas)
    weights = [float(x) for x in splitstrip(weights)]
    return datas, weights
