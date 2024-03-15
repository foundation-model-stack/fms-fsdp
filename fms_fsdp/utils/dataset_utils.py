import csv
import logging
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pyarrow as pa
import torch
import torch.utils.data as data


"""
The following distributed dataloaders are designed around 3 main principles:

1. Efficient, asynchronous operation. Workers on different devices do not communicate. 
2. Modularity. Data loading pipeline is composed of wrapped iterators, the base iterator loading from disk and additional layers 
    adding levels of post-processing (shuffling, packing, padding, etc.).
3. Seamless resumption from checkpoint. Each stage of the pipeline maintains an internal state that can be written/read on disk
    via implemented recursive `state_dict()` and `load_state_dict()` calls.
4. Rescalability. Users can save and load checkpoints to/from different numbers of workers without losing the global state. This
    is accomplished by splitting state fields for each layer into `state_params`, which are typically scalar-valued and can be
    discarded when rescaling (i.e. counters, RNG states), and `reshard_params`, which are lists that can be re-distributed over
    workers (i.e. buffers).

Our loaders obey the following type heirarchy: torch.data.IterableDataset -> _Stateful_Dataset -> _Wrapper_Dataset. `_Stateful_Dataset`
implements state and checkpointing logic. A `_Wrapper_Dataset` holds a single `_Stateful_Dataset` and iterates via calling its wrapped 
dataset any number of times, then applying some sort of post-processing and yielding the result. Users build data processing pipelines
by wrapping a base `_Stateful_Dataset` in any number of `_Wrapper_Dataset` layers, which is then passed to the torch DataLoader.

NOTE: `_Wrapper_Dataset` currently only implements wrapping a single instantiated sub-dataset layer. Many layers need multiple sub-layers
(i.e. random sampling from distinct data sources). These are currently implemented as base `_Stateful_Datasets` that take the class of 
their sub-layers plus any pass-through arguments, and instantiate all those sub-layers. This is easy on the user, who no longer needs to 
instantiate large sets of sub-layers in their code, but leads to awkwardness in this file. Cleanup is planned for the future. 
"""


def _shard_partition(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    Partition itemlist into worldsize chunks, grab chunk corresponding to rank and return.
    """
    return itemlist[
        (rank * len(itemlist)) // worldsize : ((rank + 1) * len(itemlist)) // worldsize
    ]


def _shard_inclusive(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    In cases where len(itemlist) % worldsize != 0, allow for fractional ownership of items,
    and return the span including all owned items, fractional or otherwise.
    """
    start = math.floor(len(itemlist) * rank / worldsize)
    end = math.ceil(len(itemlist) * (rank + 1) / worldsize)
    return itemlist[start:end]


class _Stateful_Dataset(data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the params to be considered stateful or reshardable in the
    self.state_params and self.reshard_params lists.
    """

    def __init__(
        self,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert (
            worldsize > rank
        ), f"Worldsize {worldsize} must be greater than rank {rank}"
        self.state_params: List[str] = []
        self.reshard_params: List[str] = []
        self.rank = rank
        self.worldsize = worldsize
        self.load_worldsize = (
            worldsize  # Enable calling load_state_dict() directly, assume no rescaling
        )

    def statename(self, x: str):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        return self.__class__.__name__ + "." + x

    def state_dict(self):
        """
        Retrieve all state and reshard flags (each worker/process saves its own state dict shard)
        """
        return {
            self.statename(flag): getattr(self, flag)
            for flag in self.state_params + self.reshard_params
        }

    def _reshard(self, sharded_list):
        """
        Sharded_list is a list of lists, where each "shard" sublist must have the same length.
        These shards should tightly span only the partition of data owned by this worker.
        (i.e. if global_list is the list of all entries, sharded_list = _shard_inclusive(global_list) ).
        Determine fractional ownership of shards, and pull out the flattened partition owned by this worker.
        """
        # How many shards did _shard_inclusive() drop to the left of sharded_list?
        shard_offset = math.floor(self.load_worldsize * self.rank / self.worldsize)
        # How long are the list shards?
        shard_len = len(sharded_list[0])
        for i, shard in enumerate(sharded_list):
            assert (
                len(shard) == shard_len
            ), f"Shard {i} with length {len(shard)} does not match expected {shard_len}"
        # How many list items did _shard_inclusive() drop to the left of the flattened sharded_list?
        item_offset = shard_len * shard_offset
        # How many list items are there in total?
        n_items = self.load_worldsize * shard_len
        # The indices of the flattened sharded_list that this worker owns
        my_items = range(
            int(n_items * self.rank / self.worldsize) - item_offset,
            int(n_items * (self.rank + 1) / self.worldsize) - item_offset,
        )
        # Pull out owned items
        return [sharded_list[i // shard_len][i % shard_len] for i in my_items]

    def load_state_dict(self, state_dicts, sharded_input=False):
        """
        Input state_dicts is a list of state_dicts. If sharded_input=False, this is expected to be the global list of states across
        all checkpoint shard files. If sharded_input=True, this expects _shard_inclusive(global_state_list).
        Handling of reduced inputs allows for much more efficient checkpoint loading.
        Workflow:
        1. if sharded_inputs is false, shard the inputs.
        2. If worldsize matches checkpoint, pull state and reshard params from the given checkpoint shard (state_dicts is a singleton list).
        3. If worldsize does not match checkpoint, toss state params and assemble reshard params from across given state_dicts.
           In this case state_dicts may be singleton (for fractional ownership) or multi-element (for multiple/partitioned ownership).
        4. Return reduced input for use by downstream loading functions
        """
        if not sharded_input:
            self.load_worldsize = len(state_dicts)
            state_dicts = _shard_inclusive(state_dicts, self.rank, self.worldsize)
        if self.load_worldsize == self.worldsize:
            [
                setattr(self, flag, state_dicts[0][self.statename(flag)])
                for flag in self.state_params + self.reshard_params
            ]
        else:
            for flag in self.reshard_params:
                reshard = self._reshard(
                    [sd[self.statename(flag)] for sd in state_dicts]
                )
                setattr(self, flag, reshard)
        return state_dicts

    def load_from_path(self, path: str):
        """
        Count shard files in the specified checkpoint folder and determine overlap with current rank and worldsize partition.
        Load only matching shardfile(s) and pass to load_state_dict. This is more efficient than sharding the full loaded state.
        """
        assert os.path.exists(path), "Specified checkpoint does not exist"
        assert not os.path.isfile(path), "Checkpoint should be a folder of shard states"
        fileshards = [x for x in os.listdir(path) if "loader" in x]
        fileshards = sorted(fileshards, key=lambda x: int(x.split("_")[2][:-4]))
        assert (
            len(fileshards) > 0
        ), "Checkpoint directory must contain checkpoint files with 'loader' in the name"
        self.load_worldsize = len(fileshards)
        # Grab only the shard files holding data we currently own
        my_fileshards = _shard_inclusive(fileshards, self.rank, self.worldsize)
        states = [torch.load(os.path.join(path, x)) for x in my_fileshards]
        self.load_state_dict(states, True)

    def save_to_path(self, path: str):
        """
        Grab recursive shard states and save all shard states to the specified checkpoint folder
        """
        os.makedirs(path, exist_ok=True)
        state = self.state_dict()
        torch.save(state, os.path.join(path, f"loader_state_{self.rank}.pth"))


class _Wrapper_Dataset(_Stateful_Dataset):
    """
    Stub for nested wrappers of _Stateful_Datasets. Extends state fns with recursion. Requires a single instantiated sub-dataset.
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
    ):
        self.dataset = dataset
        super().__init__(self.dataset.rank, self.dataset.worldsize)

    def load_state_dict(self, state_dicts, sharded_input=False):
        """
        Sets all specified flags at the current level, then recurses into wrapped dataset.
        """
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        self.dataset.load_worldsize = self.load_worldsize
        self.dataset.load_state_dict(sharded_dicts, True)
        return sharded_dicts

    def state_dict(self):
        """
        Fetches state dict recursively from wrapped layers, then adds specified flags.
        Overlapping flags are overwritten with a warning.
        """
        out = self.dataset.state_dict()
        state = super().state_dict()
        for flag in self.state_params + self.reshard_params:
            if flag in out:
                logging.warning(
                    f"Loader {self.rank}: flag {flag} already present in state_dict with value {out[flag]}. "
                    + f"Overwriting with value {state[flag]}"
                )
        out.update(state)
        return out


class Preprocess_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a _Stateful_Dataset that applies a specified preprocessing or augmentation function to dataset outputs.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    aug_fn : function (any -> any)
        The augmentation function to apply to each dataset item.
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
        aug_fn: Callable,
    ):
        super().__init__(dataset)
        self.aug_fn = aug_fn

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            yield self.aug_fn(out)


class Preload_Buffer_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a Stateful_Dataset that implements data shuffling via a single in/out buffer.
    Fills buffer two at a time, up to desired size, then switches to one at a time to maintain size.
    Passes randomly sampled outputs one by one.
    Ensures local mixing of data without relying on sliding windows or shuffling of large buffers.
    Any two consecutive inputs will be separated by window_size steps in expectation.
    Rescaling-enabled.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    window_size : int
        Max size of input/output buffer
    """

    def __init__(self, dataset: _Stateful_Dataset, window_size: int):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        self.window_size = window_size
        self.g_state = None
        self.generator = torch.Generator().manual_seed(self.rank)
        self.buffer: List[str] = []
        self.state_params = ["g_state"]
        self.reshard_params = ["buffer"]

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            # Load a point to buffer if necessary
            if len(self.buffer) <= self.window_size:
                self.buffer.append(next(dataset))

            # Load another point to buffer if necessary
            if len(self.buffer) < self.window_size:
                self.buffer.append(next(dataset))

            # Pop randomly sampled value from buffer
            i = torch.randint(len(self.buffer), (1,), generator=self.generator).item()
            yield self.buffer.pop(i)

    def state_dict(self):
        """Write generator state manually before standard behavior."""
        self.g_state = self.generator.get_state()
        return super().state_dict()

    def load_state_dict(self, state_dicts, sharded_input=False):
        """Standard load, then manually set generator state if it exists."""
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        if self.g_state is not None:
            self.generator.set_state(self.g_state)
        return sharded_dicts


class Buffer_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a _Stateful_Dataset that takes in sequences of varying lengths, and packs/pads them into sequences of desired length.
    Input sequences are packed greedily until the buffer would otherwise overrun, then remaining values are filled depending on initialization flags.
    Also injects BOS/EOS into the output sequence if desired, and if BOS/EOS tokens are not already in those positions.
    Implements rescaling by simply dropping (buffer) state.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    seq_len : int
        The desired sequence length
    pack_hard : bool
        Split input sequences to fill output buffer completely, or use pad tokens to fill remaining space?
    bos_token : any | None
        Token to add at beginning of every output sequence. If None, no token is added. Type should match data type.
    eos_token : any | None
        Token to add at end of every output sequence. If None, no token is added. Type should match data type.
    pad_token : any | None
        Token used to fill out output sequence. Type should match data type.
    drop_final_token : any | None
        Drop the final token of each document if it matches this value?
        (For edge case where bos=eos=None, and sep already appears at beginning of each doc - drop added extra sep from end of doc)
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
        seq_len: int,
        pack_hard: bool,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        drop_final_token=None,  # one-off fix for Llama training (sep already in data)
    ):
        super().__init__(dataset)
        self.len = seq_len

        # Buffer args
        self.buffer: List[str] = []
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token
        self.pack_hard = pack_hard
        if not pack_hard:
            assert (
                pad_token is not None
            ), "Error: if using pads, you must supply a pad_token"
        self.drop = drop_final_token

        self.state_params = ["buffer"]

    def _get_buffer(self, iterable, length, buffer):
        # Pull data until buffer is about to overrun, return exactly proper length
        new = []
        while len(buffer) + len(new) < length:
            buffer += new
            new = next(iterable)
            if new[-1] == self.drop:
                new = new[:-1]

        # Add bos if needed
        if self.bos is not None and (len(buffer) == 0 or buffer[0] != self.bos):
            buffer = [self.bos] + buffer

        # Handle buffer splitting
        if len(buffer) >= length:
            # If buffer is too long, force split
            out = buffer[:length]
            buffer = buffer[length:]
            if self.eos is not None and out[-1] != self.eos:
                buffer = [out[-1]] + buffer
                out[-1] = self.eos
            buffer = buffer + new
        else:
            if self.pack_hard:
                # Pack in as much of new sequence as will fit
                buffer = buffer + new
                out = buffer[:length]
                buffer = buffer[length:]
                if self.eos is not None and out[-1] != self.eos:
                    buffer = [out[-1]] + buffer
                    out[-1] = self.eos
            else:
                # Fill out with pads as needed
                if self.eos is not None and buffer[-1] != self.eos:
                    buffer.append(self.eos)
                if self.pad is not None:
                    out = buffer + [self.pad] * (length - len(buffer))
                else:
                    out = buffer
                buffer = new
        return out, buffer

    # Fill buffer line by line, delimiters and packing/splitting as appropriate
    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out, buffer = self._get_buffer(dataset, self.len, self.buffer)
            self.buffer = buffer
            yield out


class Streaming_Doc_Dataset(_Stateful_Dataset):
    """
    The base distributed dataset for loading sequences/documents from pyarrow shards.
    Pyarrow shard files are expected to hold multiple recordBatches, where each recordBatch has a "tokens" field consisting of a single token list.
    (i.e. each document is a single sequence under a "token" field, and the file is a list of such sequences)
    Relies on a compiled metadata file to fetch shardfile lengths, assumes file already exists and is in proper csv format
    (first row "dataset/filename,documents,tokens", subsequent rows these values).

    For each subdataset, splits shard files into x=worldsize fragments and grabs a 1/n contiguous span of shard fragments
    (contiguous to limit file reads from cloud/disk).
    For each section of each owned shardfile, shuffles documents and constructs an oversample list.
    Compiles oversample lists across subdatasets, and shuffles those lists deterministically, then flattens.

    At runtime, iterates through documents in each shuffled shard file, pulling each shard on demand.
    Shards are thus pulled no more than [oversample] times.
    State consists of position indices in the global shuffled oversampled doc list.
    Returns documents in chunks up to size max_chunksize, and handles delimiter token placement between documents.

    Streaming_Doc_Dataset uses integer weights to implement dataset weighting via oversampling per-epoch. For non-epoch, percentage-based
    sampling, see Sampling_Dataset, which overrides this logic.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects subfolders containing pyarrow shardfiles.
    rank : int
        Current worker index
    worldsize : int
        Total number of workers
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    trainsplit : float
        The fraction of data to assign to a training split (vs val split)
    is_val : bool
        Draw from the val split (vs from the train split)?
    datasets : list[str] | None
        A list of subdatasets to draw from. If None, draws from all subfolders.
    weights : list[int] | None
        A list of oversample rates for each subdataset. If None, draws from all subdatasets equally.
    seed : int
        The random seed for deterministic shuffling/sharding
    min_length : int
        Sequences below this length are skipped
    max_chunksize : int
        Maximum sequence length to return. Break long docs into chunks of this size or shorter.
    verbose : bool
        Track setup progress?
    shuffle : bool
        Shuffle shard file and document orders? (Disable for simple testing)
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        trainsplit: float = 1,
        is_val: bool = False,
        datasets: Optional[List[str]] = None,
        weights: Optional[List[int]] = None,
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        verbose: bool = False,
        shuffle: bool = True,
    ):
        super(Streaming_Doc_Dataset, self).__init__(rank, worldsize)
        assert (
            trainsplit >= 0 and trainsplit <= 1
        ), "Fraction of data (trainsplit) must be a positive fraction greater than 1"
        self.seed = seed
        self.data = datapath
        self.min_length = min_length
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize
        self.delimiter = delimiter_token
        self.verbose = verbose
        self.docset = []  # map of doc indices to (dataset, shardid, docid)
        self.fileset = []
        self.datasets = (
            datasets
            if datasets is not None
            else [
                f
                for f in os.listdir(datapath)
                if not os.path.isfile(os.path.join(datapath, f)) and "meta" not in f
            ]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"
        self.docs_per_dataset = {}

        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Oversample rate {w} must be a positive integer"
        self.weights = (
            {self.datasets[i]: weights[i] for i in range(len(self.datasets))}
            if weights is not None
            else {d: 1 for d in self.datasets}
        )
        # Guaranteed inconsistent shuffling across workers
        random.seed(self.seed + rank)

        # Gather per-file document counts from metadata count file(s)
        countfiles = [
            x
            for x in os.listdir(os.path.join(datapath, "meta"))
            if "counts" in x and "csv" in x
        ]
        assert len(countfiles) == 1
        doc_counts = {}
        with open(os.path.join(datapath, "meta", countfiles[0]), "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                fullpath = row["dataset/filename"]
                prefix = max([fullpath.find("/" + d) for d in self.datasets]) + 1
                if prefix > 0:
                    key = fullpath[prefix:]
                    doc_counts[key] = int(row["documents"])
        shardcount = -1

        # Assemble document sets owned by this worker
        for dataset in self.datasets:
            docset = []

            # Listdir, assemble shardfraglist (ind -> shard, frag)
            shards = [
                shard
                for shard in os.listdir(os.path.join(datapath, dataset))
                if os.path.isfile(os.path.join(datapath, dataset, shard))
                and "arrow" in os.path.join(datapath, dataset, shard)
            ]
            shards.sort()  # Ensure consistent sharding across machines
            start_frag = (rank * worldsize * len(shards)) // worldsize
            end_frag = ((rank + 1) * worldsize * len(shards)) // worldsize
            shardfrags = [
                (shards[i // worldsize], i % worldsize)
                for i in range(start_frag, end_frag)
            ]

            # Read shardfrags, assemble doc list for each file shard (aggregating over fragments):
            last_shard = ""
            ndocs = -1
            # Shardset holds all docs in this shard file section as (dataset, shardfile index, doc index)
            shardset: List[Any] = []
            for i, (shard, frag) in enumerate(shardfrags):
                # On new shard, wrap up shardset
                if shard != last_shard:
                    if len(shardset) > 0:
                        docset.append(shardset)
                    ndocs = doc_counts[os.path.join(dataset, shard)]
                    self.fileset.append(shard)
                    shardcount += 1
                    last_shard = shard
                    shardset = []

                doc_start = (ndocs * frag) // worldsize
                doc_end = (ndocs * frag + ndocs) // worldsize
                # Read into shardset
                shardset += [
                    (dataset, shardcount, i) for i in range(doc_start, doc_end)
                ]
            # Add final shardset
            if len(shardset) > 0:
                docset.append(shardset)

            # Shuffle docs inside each shardset, partition docs into train/val
            docset_slim = []
            for shardset in docset:
                shardset.sort()  # Tie partition directly to chosen seed, ignore order from keys()
                if shuffle:
                    random.shuffle(shardset)
                cutoff = math.ceil(trainsplit * len(shardset))  # Cutoff rounds up
                if is_val:
                    shardset = shardset[cutoff:]
                else:
                    shardset = shardset[:cutoff]
                docset_slim.append(shardset)
            docset = docset_slim

            # Build temp docset with oversample, add to global docset
            doccount = 0
            for shardset in docset:
                doccount += len(shardset) * self.weights[dataset]
                self.docset += [shardset] * self.weights[dataset]
            self.docs_per_dataset[dataset] = doccount

            if verbose:
                logging.info(
                    f"    Worker {rank} ingested {len(shardfrags)} shard fragments from {dataset}"
                )

        # Shuffle shardsets across datasets, and flatten
        if shuffle:
            random.shuffle(self.docset)
        self.docset = [key for shardset in self.docset for key in shardset]

        self.docset_index = 0
        self.chunk_index = -1

        # Stats
        self.epochs_seen = -1
        self.dataset_tokens_seen = {d: 0 for d in self.datasets}
        self.dataset_docs_seen = {d: 0 for d in self.datasets}
        self.dataset_percent_seen = {d: 0 for d in self.datasets}
        # self.docs_seen: Dict[Any, int] = {}  # (dataset, shard, i) -> # times seen

        self.state_params = [
            "docset_index",
            "chunk_index",
            "epochs_seen",
            "dataset_tokens_seen",
            "dataset_docs_seen",
            "dataset_percent_seen",
            # "docs_seen",
        ]

    def get_reader(self, path, newpath, reader):
        """
        If new filepath does not match the current one, open a new reader on that filepath (pull file on demand)
        """
        if newpath != path:
            if self.verbose:
                logging.info(f"Worker {self.rank} opening new file {newpath}")
            reader = pa.ipc.open_file(newpath)
            path = newpath
        return path, reader

    def _construct_chunk(self, j, doc, n_chunks, dataset):
        """
        Grab a chunk of the desired size from the pyarrow document, avoiding unnecessary overhead in case of large docs
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        chunk = doc.slice(start_index, n_pull).to_pylist()
        self.dataset_tokens_seen[dataset] += len(chunk)
        if j == n_chunks - 1:
            chunk = chunk + [
                self.delimiter
            ]  # Add delimiter token to signify end of document (used upstream)
        return chunk

    def __iter__(self):
        docset_offset = self.docset_index
        residual_chunks = self.chunk_index + 1  # pick up AFTER where the ckp left off
        ndocs = len(self.docset)
        path = ""
        reader = None
        while True:
            # Iterate through docs, starting at desired offset
            for i in range(ndocs):
                doc_index = (docset_offset + i) % ndocs

                # Update stats
                if doc_index == 0:
                    self.epochs_seen += 1
                self.docset_index = doc_index
                key = self.docset[doc_index]
                # if key in self.docs_seen:
                #     self.docs_seen[key] += 1
                # else:
                #     self.docs_seen[key] = 1
                dataset, shardid, docid = key
                self.dataset_docs_seen[dataset] += 1
                self.dataset_percent_seen[dataset] = (
                    self.dataset_docs_seen[dataset]
                    * 100
                    / (self.docs_per_dataset[dataset] + 1e-9)
                )

                # Read doc
                newpath = os.path.join(self.data, dataset, self.fileset[shardid])
                path, reader = self.get_reader(path, newpath, reader)
                doc = reader.get_batch(docid)["tokens"]
                if len(doc) >= self.min_length:
                    n_chunks = math.ceil(
                        (len(doc) + 1) / self.chunksize
                    )  # add 1 for eos
                    for j in range(n_chunks):
                        if i == 0 and j < residual_chunks:
                            pass
                        else:
                            self.chunk_index = j
                            yield self._construct_chunk(j, doc, n_chunks, dataset)

            # Load any chunks initially skipped in first doc
            self.docset_index = docset_offset
            key = self.docset[docset_offset]
            dataset, shardid, docid = key
            newpath = os.path.join(self.data, dataset, self.fileset[shardid])
            path, reader = self.get_reader(path, newpath, reader)
            doc = reader.get_batch(docid)["tokens"]  # .to_pylist()
            if len(doc) >= self.min_length:
                n_chunks = math.ceil((len(doc) + 1) / self.chunksize)  # add 1 for eos
                for j in range(residual_chunks):
                    self.chunk_index = j
                    yield self._construct_chunk(j, doc, n_chunks, dataset)

    def load_state_dict(self, state_dicts, sharded_input=False):
        assert (
            self.load_worldsize == self.worldsize
        ), "Streaming_Doc_Dataset does not support rescaling"
        return super().load_state_dict(state_dicts, sharded_input)


class Sampling_Dataset(_Stateful_Dataset):
    """
    A _Stateful_Dataset implementing percentage-based sampling: weights can be floats and the number of tokens seen from each subdataset will match those weights as closely as possible.
    This is accomplished by maintaining a _Stateful_Dataset for each subdataset, and tracking the number of tokens emitted by each.
    Whichever loader is furthest from its target will be the next to pass a document.
    All args except for dataset and weights are pass-through args for the component _Stateful_Datasets and are documented in the appropriate classes.
    ...
    Args
    ----
    dataset : Scalable_Shard_Dataset | Streaming_Doc_Dataset
        Underlying iterator for each desired subdataset
    weights : list(float) | None
        Weights describing what percent of emitted tokens should come from each subdataset. Need not sum to 1. If None, tokens are drawn evenly.
    ...
        Pass-through args, see Streaming_Doc_Dataset or Scalable_Shard_Dataset
    """

    def __init__(
        self,
        datapath: str,
        dataset: Union[
            Type["Streaming_Doc_Dataset"],
            Type["Scalable_Shard_Dataset"],
        ],
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        datasets=None,
        weights=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(rank, worldsize)
        self.delimiter = delimiter_token
        self.datasets = (
            datasets
            if datasets is not None
            else [
                f
                for f in os.listdir(datapath)
                if not os.path.isfile(os.path.join(datapath, f)) and "meta" not in f
            ]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"

        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Sampling rate {w} must be positive"
        self.weights = [1] * len(self.datasets) if weights is None else weights
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.tokens_seen = [0] * len(self.datasets)

        # Build subdataset iterators
        self.data = []
        for i, d in enumerate(self.datasets):
            self.data.append(
                dataset(
                    datapath=datapath,
                    rank=rank,
                    worldsize=worldsize,
                    delimiter_token=delimiter_token,
                    weights=None,
                    datasets=[d],
                    verbose=verbose,
                    **kwargs,
                )
            )
            if verbose:
                logging.info(
                    f"Worker {rank} assembled subdataset iterator for {d}, {i+1} of {len(self.datasets)}"
                )

        self.current_iterator = -1
        self.state_params = ["tokens_seen", "current_iterator"]

    def __iter__(self):
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            if self.current_iterator != -1:
                # Finish current document
                out = next(data[self.current_iterator])
                self.tokens_seen[self.current_iterator] += len(out)
                if out[-1] == self.delimiter:
                    self.current_iterator = -1
                yield out
            else:
                # Choose new subdataset to draw from
                # (whichever is currently most underrepresented compared to target rate)
                offset = [
                    self.weights[i]
                    - self.tokens_seen[i] / (sum(self.tokens_seen) + 1e-9)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = max((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax

    def state_dict(self):
        # Manually add state of all subloaders to self state
        out = {
            self.statename("sample_iterator_states"): [
                d.state_dict() for d in self.data
            ]
        }
        out.update(super().state_dict())
        return out

    def load_state_dict(self, state_dicts, sharded_input=False):
        # Load stats
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        # Load sub-iterator states
        for i, subdata in enumerate(self.data):
            # Grab just that sub-iterator across all ranks
            subdata.load_worldsize = self.load_worldsize
            subdata.load_state_dict(
                [
                    sd[self.statename("sample_iterator_states")][i]
                    for sd in sharded_dicts
                ],
                True,
            )
        return sharded_dicts


class Scalable_Shard_Dataset(_Stateful_Dataset):
    """
    A _Stateful_Dataset implementing rescalability: loading from checkpoint into a different number of gpus will nonetheless keep avoiding all data previously seen in the current epoch.
    This is accomplished by maintaining a large number of small Streaming_Doc_Datasets, which track state individually and reshard over n_gpus.
    All keywords except the first are simple pass-through arguments and are documented in Streaming_Doc_Dataset.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects subfolders containing pyarrow shardfiles.
    rank : int
        Current worker index
    worldsize : int
        Total number of workers
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    n_logical_shards : int
        Number of logical shards. Must be a multiple of world size.
    ...
        Pass-through args, see Streaming_Doc_Dataset
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        n_logical_shards: int = 2048,
        verbose=False,
        **kwargs,
    ):
        assert (
            n_logical_shards % worldsize == 0
        ), f"World size {worldsize} must divide n_logical_shards {n_logical_shards} evenly"
        assert (
            n_logical_shards > 0
        ), f"n_logical_shards {n_logical_shards} must be a positive integer"

        super().__init__(rank, worldsize)
        self.data = []
        self.docset: List[Any] = []
        self.n_logicals = n_logical_shards // worldsize
        self.total_shards = n_logical_shards
        self.delimiter = delimiter_token

        logicals = list(range(n_logical_shards))
        self.logicals_owned = _shard_partition(logicals, self.rank, self.worldsize)
        assert len(self.logicals_owned) == self.n_logicals

        # Build logical shards
        for i in range(self.n_logicals):
            self.data.append(
                Streaming_Doc_Dataset(
                    datapath=datapath,
                    worldsize=n_logical_shards,
                    rank=self.logicals_owned[i],
                    delimiter_token=delimiter_token,
                    verbose=(rank == 0),
                    **kwargs,
                )
            )
            if verbose:
                logging.info(
                    f"Worker {rank} assembled logical shard {self.logicals_owned[i]}, {i+1} of {self.n_logicals}"
                )

        # Fetch logical shard sampling stats
        self.n_docs_remaining = [len(d.docset) for d in self.data]

        # Position "state", used only for maintaining order when n_workers is unchanged
        # For scaling up or down, logical position is meaningless, and reset
        self.current_reader = None
        self.shuffle_state = self.rank
        self.logical_shard_states = None
        self.state_params = ["current_reader", "shuffle_state"]
        self.reshard_params = ["n_docs_remaining", "logical_shard_states"]

    def __iter__(self):
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            # Sample logical shard (or load from ckp)
            if self.current_reader is not None:
                ind = self.current_reader
            else:
                random.seed(self.shuffle_state)
                ind = random.choices(
                    list(range(self.n_logicals)), weights=self.n_docs_remaining, k=1
                )[0]
                self.shuffle_state = (self.shuffle_state + 1) % 10000
            self.current_reader = ind
            # Read doc
            out = next(data[ind])
            while out[-1] != self.delimiter:
                yield out
                out = next(data[ind])
            # Update state to show we've finished the doc
            self.current_reader = None
            self.n_docs_remaining[ind] -= 1
            if sum(self.n_docs_remaining) == 0:
                self.n_docs_remaining = [len(d.docset) for d in self.data]
            # Return final piece of doc
            yield out

    def state_dict(self):
        self.logical_shard_states = [d.state_dict() for d in self.data]
        return super().state_dict()

    def load_state_dict(self, state_dicts, sharded_input=False):
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        for i in range(self.n_logicals):
            self.data[i].load_state_dict([self.logical_shard_states[i]], True)
        return sharded_dicts
