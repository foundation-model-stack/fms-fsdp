import functools
import os
import tempfile
from collections import Counter
from copy import deepcopy
from itertools import chain

import pyarrow as pa
import torch

from fms_fsdp.utils.dataset_utils import *


# Generates test data in a temp directory, and returns that tempdir object.
# (file path can be retrieved via tempdir.name)
# Two dataset folders: one has a large shardfile (100x100), other has two small shardfiles (50x50)
def generate_sequential_multidata():
    tmpdir = tempfile.TemporaryDirectory()
    schema = pa.schema([pa.field("tokens", pa.uint32())])

    os.mkdir(os.path.join(tmpdir.name, "dataset_1"))
    os.mkdir(os.path.join(tmpdir.name, "dataset_2"))
    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_1/fullshard.arrow"), schema
    ) as writer:
        for i in range(100):
            out = list(range(i * 100, i * 100 + 100))
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/quartershard_1.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(i * 50, i * 50 + 50))
            writer.write(pa.record_batch([out], schema=schema))

    with pa.ipc.new_file(
        os.path.join(tmpdir.name, "dataset_2/quartershard_2.arrow"), schema
    ) as writer:
        for i in range(50):
            out = list(range(2500 + i * 50, 2500 + i * 50 + 50))
            writer.write(pa.record_batch([out], schema=schema))

    # Make metadata file
    os.mkdir(os.path.join(tmpdir.name, "meta"))
    f = open(os.path.join(tmpdir.name, "meta", "combined_counts.csv"), "w")
    f.write("dataset/filename,documents,tokens\n")
    f.write("/dataset_1/fullshard.arrow,100,10000\n")
    f.write("/dataset_2/quartershard_1.arrow,50,2500\n")
    f.write("/dataset_2/quartershard_2.arrow,50,2500\n")
    f.close()

    return tmpdir


# Make mock data for re-use. Returns directory path.
tmpdir = generate_sequential_multidata()


# REPEATED CHECKS
# Checks take a dataset definition (and any other args), instantiate it, and perform a single unit test
# For X_check see corresponding test_X


def count_check(d, dataset, ntok, ndoc, alldoc, allpercent):
    # Check that tokens tracked matches tokens seen, and docs tracked matches docs seen
    # d is a lambda for a fully-defined dataset (i.e. d() instantiates the dataset)
    assert (
        d.dataset_tokens_seen[dataset] == ntok
    ), f"Tokens tracked {d.dataset_tokens_seen[dataset]} failed to match target {ntok}"
    # for k in d.docs_seen.keys():
    #     if k[0] == dataset:
    #         assert d.docs_seen[k] == ndoc, f"Document {k} appeared {d.docs_seen[k]} times rather than {ndoc}"
    # assert (
    #     d.dataset_docs_seen[dataset] == alldoc
    # ), f"Total document count {d.dataset_docs_seen[dataset]} does not match target {alldoc}"
    coverage = d.dataset_percent_seen[dataset]
    assert (
        abs(coverage - allpercent) < 1e-4
    ), f"Percent coverage {coverage} is not within 1e-4 of {allpercent}"


def multi_reload_stress_check(d):
    # Perform the reload stress test for different numbers of steps before and after checkpoint
    # d is a lambda for a fully-defined dataset (i.e. d() instantiates the dataset)

    def reload_stress(datasets, datasets2, steps1, steps2):
        # Perform the 5-step reload stress test (see test_reload_stress_all)

        loaders = [iter(d) for d in datasets]

        for i in range(steps1):
            [next(l) for l in loaders]

        states = [deepcopy(d.state_dict()) for d in datasets]

        [d.load_state_dict(states) for d in datasets2]

        loaders2 = [iter(d) for d in datasets2]

        for k in range(steps2):
            for i in range(3):
                out1 = next(loaders[i])
                out2 = next(loaders2[i])
                assert len(out1) == len(
                    out2
                ), f"Dataloader {i} in step {k} has mismatched length: {len(out1)} vs {len(out2)}"
                for j in range(len(out1)):
                    assert (
                        out1[j] == out2[j]
                    ), f"Dataloader {i} in step {k} has mismatched token in position {j}: {out1[j]} vs {out2[j]}"

    steps1 = [0, 1, 10, 100, 1000]
    steps2 = [100, 200, 300, 400, 500]
    for i in range(len(steps1)):
        # Reset between tests (instantiate fresh datasets)
        reload_stress(d(), d(), steps1[i], steps2[i])


def single_epoch_check(d, do_countcheck=False):
    # For a single loader on dataset_1, check that every doc appears once per epoch
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset = d(datasets=["dataset_1"])
    ins = []
    loader = iter(dataset)
    for i in range(100):
        out = next(loader)
        ins.append(out[0])

    for i in range(100):
        assert (
            i * 100 in ins
        ), f"Line starting with {i * 100} failed to appear in generated data"

    if do_countcheck:
        # Check state flags tracking correctly
        count_check(dataset, "dataset_1", 100 * 100, 1, 100, 100)


def two_epoch_check(d, do_countcheck=False):
    # For a single loader on dataset_1, check that every doc appears twice per two epochs
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset = d(datasets=["dataset_1"])
    ins = []
    loader = iter(dataset)
    for i in range(100 * 2):
        out = next(loader)
        ins.append(out[0])

    for i in range(100):
        key = ins.pop(0)
        assert (
            key in ins
        ), f"Line starting with {key} failed to appear a second time in generated data"

    if do_countcheck:
        # Check state flags tracking correctly
        count_check(dataset, "dataset_1", 100 * 100 * 2, 2, 200, 200)


def chunk_check(d, do_countcheck=False):
    # For a single loader on dataset_1, check that every doc chunks properly and that all chunks appear in one epoch
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset = d(datasets=["dataset_1"], max_chunksize=50)
    ins = []
    loader = iter(dataset)
    for i in range(300):
        out = next(loader)
        if i % 3 != 2:
            assert (
                len(out) == 50
            ), f"Line length {len(out)} does not match chunk size 50"
        else:
            assert (
                out[0] == -1
            ), f"Chunk 3 of document {i} is not delimiter token, but {out}"
        ins.append(out[0])

    for i in range(200):
        assert (
            i * 50 in ins
        ), f"Chunk starting with {i * 50} failed to appear in generated data"

    if do_countcheck:
        count_check(dataset, "dataset_1", 100 * 100, 1, 100, 100)


def two_loader_check(d, do_countcheck=False):
    # For two loaders on dataset_1, check that every doc appears once per epoch, collectively
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset1 = d(datasets=["dataset_1"], worldsize=2, rank=0)
    dataset2 = d(datasets=["dataset_1"], worldsize=2, rank=1)
    ins = []
    loader = iter(dataset1)
    for i in range(50):
        out = next(loader)
        ins.append(out[0])
    loader = iter(dataset2)
    for i in range(50):
        out = next(loader)
        ins.append(out[0])

    for i in range(100):
        assert (
            i * 100 in ins
        ), f"Line starting with {i * 100} failed to appear in generated data"

    if do_countcheck:
        count_check(dataset1, "dataset_1", 50 * 100, 1, 50, 100)
        count_check(dataset2, "dataset_1", 50 * 100, 1, 50, 100)


def multi_file_check(d, do_countcheck=False):
    # For a single loader on dataset 2, check that every doc appears once per epoch
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset = d(datasets=["dataset_2"])
    ins = []
    loader = iter(dataset)
    for i in range(100):
        out = next(loader)
        ins.append(out[0])

    for i in range(100):
        assert (
            i * 50 in ins
        ), f"Line starting with {i * 50} failed to appear in generated data"

    if do_countcheck:
        count_check(dataset, "dataset_2", 100 * 50, 1, 100, 100)


def chunk_weight_check(w1, w2, d, do_countcheck=False):
    # For a single loader on combined datasets, with given oversamples, chunksize 50: check that chunks appear the proper number of times
    # d is a lambda for a dataset (i.e. d() instantiates the dataset)
    dataset = d(datasets=["dataset_1", "dataset_2"], weights=[w1, w2], max_chunksize=50)
    ins = []
    loader = iter(dataset)
    for i in range(3 * w1 * 100 + 2 * w2 * 100):
        out = next(loader)
        if len(out) > 1:
            ins.append(out[0])

    check = Counter(ins)
    for i in range(200):
        if i < 100:
            assert (
                check[i * 50] == w1 + w2
            ), f"Chunk starting with {i * 50} appeared {check[i*50]} times rather than {w1+w2}"
        else:
            assert (
                check[i * 50] == w1
            ), f"Chunk starting with {i * 50} appeared {check[i*50]} times rather than {w1}"

    # Check state flags tracking correctly
    if do_countcheck:
        count_check(dataset, "dataset_1", 100 * 100 * w1, w1, 100 * w1, 100)
        count_check(dataset, "dataset_2", 100 * 50 * w2, w2, 100 * w2, 100)


def reload_epoch_check(loader):
    # Single shard, two loaders: do exactly 1/3 of an epoch, checkpoint, reload to same number of workers.
    # Complete the epoch and verify that no loaded chunks are revisiting old chunks.
    datasets = [
        loader(
            rank=i,
            worldsize=2,
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    ins = []
    for _ in range(50):
        out = next(loaders[0])
        ins.append(out[0])
    for _ in range(50):
        out = next(loaders[1])
        ins.append(out[0])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        loader(
            rank=i,
            worldsize=2,
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    loaders2 = [iter(d) for d in datasets2]

    for j in range(100):
        for i in range(2):
            out = next(loaders2[i])
            assert (
                out[0] not in ins
            ), f"Step {j+1}, dataset {i+1}: chunk starting with {out[0]} has already appeared in the epoch"


def reload_single_epoch_check(loader):
    # Single shard, two loaders: advance 37 steps, checkpoint, reload to same number of workers.
    # Run a full epoch and verify that all data appears once and only once.
    datasets = [
        loader(
            rank=i,
            worldsize=2,
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    for _ in range(37):
        out = next(loaders[0])
    for _ in range(37):
        out = next(loaders[1])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        loader(
            rank=i,
            worldsize=2,
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    loaders2 = [iter(d) for d in datasets2]

    ins = []
    for _ in range(150):
        out = next(loaders2[0])
        ins.append(out[0])
    for _ in range(150):
        out = next(loaders2[1])
        ins.append(out[0])

    assert len(ins) == len(
        set(ins)
    ), f"Full epoch output contains {len(ins)} values but only {len(set(ins))} unique"


# BASE DATASET TESTS


def basic_loader(
    rank=0, worldsize=1, datasets=["dataset_1"], weights=[1], max_chunksize=1000
):
    return Streaming_Doc_Dataset(
        tmpdir.name,
        rank,
        worldsize,
        -1,
        datasets=datasets,
        weights=weights,
        max_chunksize=max_chunksize,
    )


def basic_sampler(
    rank=0, worldsize=1, datasets=["dataset_1"], weights=[1], max_chunksize=1000
):
    return Sampling_Dataset(
        tmpdir.name,
        Streaming_Doc_Dataset,
        rank,
        worldsize,
        -1,
        datasets=datasets,
        weights=weights,
        max_chunksize=max_chunksize,
    )


def basic_scalable(
    rank=0,
    worldsize=1,
    datasets=["dataset_1"],
    weights=[1],
    max_chunksize=1000,
    n_logical_shards=7,
):
    return Scalable_Shard_Dataset(
        tmpdir.name,
        rank,
        worldsize,
        -1,
        datasets=datasets,
        weights=weights,
        max_chunksize=max_chunksize,
        n_logical_shards=n_logical_shards,
    )


def basic_scalable_sampler(
    rank=0,
    worldsize=1,
    datasets=["dataset_1"],
    weights=[1],
    max_chunksize=1000,
    n_logical_shards=7,
):
    return Sampling_Dataset(
        tmpdir.name,
        Scalable_Shard_Dataset,
        rank,
        worldsize,
        -1,
        datasets=datasets,
        weights=weights,
        max_chunksize=max_chunksize,
        n_logical_shards=n_logical_shards,
    )


def test_single_epoch():
    # Single shard, single loader: every line appears once in an epoch
    single_epoch_check(basic_loader, True)
    single_epoch_check(basic_scalable)
    single_epoch_check(basic_sampler)
    single_epoch_check(basic_scalable_sampler)


def test_two_epoch():
    # Single shard, single loader: every line appears twice in two epochs
    two_epoch_check(basic_loader, True)
    two_epoch_check(basic_scalable)
    two_epoch_check(basic_sampler)
    two_epoch_check(basic_scalable_sampler)


def test_chunk():
    # Single shard, single loader, two chunks/doc plus a delimiter token: every chunk appears once in an epoch
    chunk_check(functools.partial(basic_loader, max_chunksize=50), True)
    chunk_check(functools.partial(basic_scalable, max_chunksize=50))
    chunk_check(functools.partial(basic_sampler, max_chunksize=50))
    chunk_check(functools.partial(basic_scalable_sampler, max_chunksize=50))


def test_two_loader():
    # Single shard, two loaders: every line appears once per epoch, collectively
    two_loader_check(basic_loader, True)
    two_loader_check(functools.partial(basic_scalable, n_logical_shards=8))
    two_loader_check(basic_sampler)
    two_loader_check(functools.partial(basic_scalable_sampler, n_logical_shards=8))


def test_multi_file():
    # Multiple shard files, single loader: every line appears once in an epoch
    multi_file_check(basic_loader, True)
    multi_file_check(basic_scalable)
    multi_file_check(basic_sampler)
    multi_file_check(basic_scalable_sampler)


def test_reload_epoch():
    # Single shard, two loaders: check that reloading mid-epoch does not cause data to repeat while finishing the epoch
    reload_epoch_check(basic_loader)
    reload_epoch_check(functools.partial(basic_scalable, n_logical_shards=8))
    reload_epoch_check(basic_sampler)
    reload_epoch_check(functools.partial(basic_scalable_sampler, n_logical_shards=8))


def test_reload_complete_epoch():
    # Single shard, two loaders: check that reloading mid-epoch can still complete a full epoch
    # Skip scalable loaders as epoch 2 may sample workers differently from epoch 1
    reload_single_epoch_check(basic_loader)
    reload_single_epoch_check(basic_sampler)


# SUBDATASET WEIGHTING CHECKS


def test_weight():
    """
    A test for base, non-wrapper datasets using oversampling (currently Streaming_Doc and Scalable_Shard).
    On the full dataset, with varying weights, on a single worker: with chunksize 50, run one epoch.
    Verify that first half of dataset_1 is replicated by dataset_2, and second half is not, with appropriate ratios.
    """
    w1 = [1, 2, 4]
    w2 = [1, 3, 7]
    for i in w1:
        for j in w2:
            # Test chunk weights on doc dataloader (single epoch)
            doc_dataset = functools.partial(
                basic_loader,
                datasets=["dataset_1", "dataset_2"],
                weights=[i, j],
                max_chunksize=50,
            )
            chunk_weight_check(i, j, doc_dataset, True)
            # Test chunk weights on scalable dataloader (single epoch)
            shard_dataset = functools.partial(
                basic_scalable,
                datasets=["dataset_1", "dataset_2"],
                weights=[i, j],
                max_chunksize=50,
            )
            chunk_weight_check(i, j, shard_dataset)


def test_sampler_rates():
    """
    A test for base, non-wrapper datasets using percentage sampling (currently Sampling and Scalable_Sampling).
    On the full dataset, with varying weights, on a single worker: verify that loaders pull subdatasets at regular intervals
    (verifying that they're regularly picking the most-underviewed subdataset at each step).
    """
    weights = [[1, 1], [2, 1], [2, 3], [2, 5]]
    target_rate = [3, 2, 4, 6]
    burnin = [3, 0, 4, 6]

    # Dataset1 docs are twice the length of dataset2. Burnin required to reach equilibrium.
    # Expected sequences for each case are:
    # 2 1 2 1 2 2 (1 2 2)...
    # 1 2 1 2 (1 2)...
    # 2 1 2 2 1 2 2 2 (1 2 2 2)...
    # 2 1 2 2 2 2 1 2 2 2 2 2 (1 2 2 2 2 2)...

    def check_rates(w, t, b, m):
        s = []
        d = m(datasets=["dataset_1", "dataset_2"], weights=w)
        l = iter(d)
        for i in range(b):
            s.append(len(next(l)))
        for i in range(100):
            out = next(l)
            s.append(len(out))
            if i % t == 0:
                assert (
                    len(out) == 101
                ), f"Output {i} length {len(out)} does not match expected 101. Sequence so far: {s}"
            else:
                assert (
                    len(out) == 51
                ), f"Output {i} length {len(out)} does not match expected 51. Sequence so far: {s}"

    for i in range(3):
        for m in [basic_sampler, basic_scalable_sampler]:
            check_rates(weights[i], target_rate[i], burnin[i], m)


# STRESS TEST


def test_multi_reload_stress():
    """
    For each nontrivial layer of the dataset pipeline:
        For each combo of steps:
            Initialize two identical datasets
            Take n steps with the first one
            Save checkpoint
            Load checkpoint into second dataset
            Take k steps with both datasets, check that outputs are identical
    Parameters are chosen to ensure messy states (non-divisible chunk sizes, shard numbers, n_workers, buffer_length, etc.)
    """
    # Shard doc dataset
    d1 = lambda: [
        Streaming_Doc_Dataset(
            tmpdir.name,
            i,
            3,
            -1,
            datasets=["dataset_1", "dataset_2"],
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    multi_reload_stress_check(d1)

    # Sampling dataset
    d2 = lambda: [
        Sampling_Dataset(
            tmpdir.name,
            Streaming_Doc_Dataset,
            i,
            3,
            -1,
            datasets=["dataset_1", "dataset_2"],
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    multi_reload_stress_check(d2)

    # Scalable shard dataset
    d3 = lambda: [
        Scalable_Shard_Dataset(
            tmpdir.name,
            i,
            3,
            -1,
            n_logical_shards=15,
            datasets=["dataset_1", "dataset_2"],
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    multi_reload_stress_check(d3)

    # Nested scalable sampling dataset
    d4 = lambda: [
        Sampling_Dataset(
            tmpdir.name,
            Scalable_Shard_Dataset,
            i,
            3,
            -1,
            n_logical_shards=15,
            datasets=["dataset_1", "dataset_2"],
            weights=[3, 5],
            max_chunksize=17,
        )
        for i in range(3)
    ]
    multi_reload_stress_check(d4)

    # Add buffer dataset
    d5 = lambda x: [Buffer_Dataset(d, 73, pack_hard=True, bos_token=-1) for d in x]
    # Stress test buffer + all bases
    for d in [d1, d2, d3, d4]:
        multi_reload_stress_check(lambda: d5(d()))

    # # Add PLM dataset
    # d6 = lambda x: [PLM_Dataset(d, -3, -2) for d in x]
    # # plm / buffer / sample / scale / doc pipeline
    # multi_reload_stress_check(lambda: d6(d5(d4())))

    # Add preload buffer dataset
    d7 = lambda x: [Preload_Buffer_Dataset(d, 99) for d in x]
    # preload / sample / scale / doc pipeline
    multi_reload_stress_check(lambda: d7(d4()))


# SCALABLE_DATASET TESTS


def test_scalable_partitioning():
    """
    Test that partitioning occurs correctly when rescaling up or down, including to non-multiples of the original
    physical worker count. Start with 4 workers with 12 logical shards, and for each of [1,2,3,6,12], verify that:
    1) no overlap exists between workers and 2) in over one epoch's worth of steps, each data point appears at least once
    """
    for layer in [Scalable_Shard_Dataset, Sampling_Dataset]:
        kwargs = {
            "n_logical_shards": 12,
            "datasets": ["dataset_1"],
            "weights": [1],
            "max_chunksize": 200,
        }
        datasets = [
            layer(tmpdir.name, Scalable_Shard_Dataset, i, 4, -1, **kwargs)
            if layer == Sampling_Dataset
            else layer(tmpdir.name, i, 4, -1, **kwargs)
            for i in range(4)
        ]  # 25 steps per epoch
        loaders = [iter(d) for d in datasets]

        for _ in range(50):
            [next(l) for l in loaders]

        states = [d.state_dict() for d in datasets]

        kwargs = {
            "n_logical_shards": 12,
            "datasets": ["dataset_1"],
            "weights": [1],
            "max_chunksize": 200,
        }
        for worldsize in [1, 2, 3, 6, 12]:
            datasets = [
                layer(
                    tmpdir.name,
                    Scalable_Shard_Dataset,
                    i,
                    worldsize,
                    -1,
                    **kwargs,
                )
                if layer == Sampling_Dataset
                else layer(
                    tmpdir.name,
                    i,
                    worldsize,
                    -1,
                    **kwargs,
                )
                for i in range(worldsize)
            ]
            [d.load_state_dict(states) for d in datasets]
            loaders = [iter(d) for d in datasets]
            outs = [[] for _ in datasets]
            steps = int(100 / worldsize * 1.25)
            for i in range(steps):
                for j, l in enumerate(loaders):
                    outs[j].append(next(l)[0])

            # Check for non-overlap
            for i in range(len(datasets)):
                for j in range(i + 1, len(datasets)):
                    outi = set(outs[i])
                    outj = set(outs[j])
                    for t in outi:
                        assert (
                            t not in outj
                        ), f"Overlapping value {t} detected in worker {i} and {j}: {outi}, {outj}"
                    for t in outj:
                        assert (
                            t not in outi
                        ), f"Overlapping value {t} detected in worker {i} and {j}: {outi}, {outj}"

            # Check for completion
            allout = set(chain(*outs))
            for i in range(100):
                assert i * 100 in allout, f"Token {i*100} missing from outputs {allout}"


def test_scalable_shard_reload_scale():
    """
    As test_reload_epoch, but in this case we scale from 2 workers to 4 (complete 1/3 epoch, reload, finish without duplication).
    Because logical shards won't all be the exact same length when checkpointed, we complete the epoch of the shortest of the new workers.
    """
    datasets = [
        Scalable_Shard_Dataset(
            tmpdir.name,
            i,
            2,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    ins = []
    for _ in range(50):
        out = next(loaders[0])
        ins.append(out[0])
    for _ in range(50):
        out = next(loaders[1])
        ins.append(out[0])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        Scalable_Shard_Dataset(
            tmpdir.name,
            i,
            4,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(4)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    ndocs = [sum(d.n_docs_remaining) for d in datasets]
    print("n_docs_remaining from old loader:", ndocs)
    ndocs = [sum(d.n_docs_remaining) for d in datasets2]
    print("n_docs_remaining per new loader:", ndocs)

    loaders2 = [iter(d) for d in datasets2]

    print("Checking only", min(ndocs) * 3, "steps instead of full 50")
    for j in range(min(ndocs) * 3):
        for i in range(4):
            out = next(loaders2[i])
            assert (
                out[0] not in ins
            ), f"Step {j+1}, dataset {i+1}: chunk starting with {out[0]} has already appeared in the epoch"


def test_scalable_sampler_reload_scale():
    """
    As test_reload_epoch, but in this case we scale from 2 workers to 4 (complete 1/3 epoch, reload, finish without duplication).
    Because logical shards and sampling ratios won't be exact, take a few extra steps then check that epoch is complete.
    """
    datasets = [
        Sampling_Dataset(
            tmpdir.name,
            Scalable_Shard_Dataset,
            i,
            2,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(2)
    ]  # Length 300
    loaders = [iter(d) for d in datasets]

    ins = []
    for _ in range(50):
        out = next(loaders[0])
        ins.append(out[0])
    for _ in range(50):
        out = next(loaders[1])
        ins.append(out[0])

    states = [d.state_dict() for d in datasets]

    datasets2 = [
        Sampling_Dataset(
            tmpdir.name,
            Scalable_Shard_Dataset,
            i,
            4,
            -1,
            n_logical_shards=8,
            datasets=["dataset_1"],
            weights=[1],
            max_chunksize=40,
        )
        for i in range(4)
    ]  # Length 300
    [d.load_state_dict(states) for d in datasets2]
    loaders2 = [iter(d) for d in datasets2]

    for i in range(4):
        for _ in range(55):
            out = next(loaders2[i])
            ins.append(out[0])

    for suf in [0, 40, 80]:
        for i in range(100):
            assert (
                i * 100 + suf in ins
            ), f"Expected value {i*100+suf} not found in output set {ins}"


# BUFFER_DATASET TESTS


class RandCounter:
    # Spit out incremental counts of random length, uniformly sampled from 1 to 50
    def __init__(self):
        self.i = 0
        self.rank = 0
        self.worldsize = 1

    def __iter__(self):
        while True:
            l = torch.randint(1, 50, [1]).item()
            yield list(range(self.i, self.i + l))
            self.i += l


def test_buffer_format():
    # Using the RandCounter, verify that streams are reformed into correct-length buffers,
    # that final tokens match the predicted count, and that BOS/EOS add correctly

    for _ in range(100):
        # 100 trials of random length inputs
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True)
        loader = iter(dataset)
        for _ in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
        assert (
            out[-1] == 100 * 100 - 1
        ), f"Final token {out[-1]} does not match expected value {100*100-1}"

    # As above, but now with EOS tokens
    for _ in range(100):
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True, eos_token=-1)
        loader = iter(dataset)
        for i in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
            assert out[-1] == -1, f"Output {out} does not end in EOS"
        assert (
            out[-2] == 100 * 99 - 1
        ), f"Penultimate token {out[-2]} does not match expected value {100*99-1}"

    # As above, but now with BOS tokens
    for _ in range(100):
        base = RandCounter()
        dataset = Buffer_Dataset(base, 100, pack_hard=True, bos_token=-1)
        loader = iter(dataset)
        for i in range(100):
            out = next(loader)
            assert (
                len(out) == 100
            ), f"Length of output {len(out)} does not match specified 100"
            assert out[0] == -1, f"Output {out} does not begin with BOS"
        assert (
            out[-1] == 100 * 99 - 1
        ), f"Final token {out[-1]} does not match expected value {100*99-1}"


def test_buffer_delimiter_overlap():
    """
    Check that BOS adds correctly when absent, and refrains when present.
    Because doc delimiter token is also -1, BOS will add in the first instance, which shunts the delimiter token
    into the first slot in the next (and all subsequent) outputs. BOS should then refrain from adding.
    """
    dataset = basic_loader(max_chunksize=101)
    dataset = Buffer_Dataset(dataset, 101, pack_hard=True, bos_token=-1)
    loader = iter(dataset)
    for _ in range(100):
        out = next(loader)
        assert (
            len(out) == 101
        ), f"Length of output {len(out)} does not match specified 101"
        assert out[0] == -1, f"Output {out} does not begin with BOS"
    assert (
        out[-1] % 100 == 99
    ), f"Final token {out[-1]} does not end in expected value 99"


# PRELOAD_BUFFER_DATASET TESTS


class SteadyCounter:
    # Spit out incremental counts of constant length l
    def __init__(self, l):
        self.i = 0
        self.rank = 0
        self.worldsize = 1
        self.l = l

    def __iter__(self):
        while True:
            yield list(range(self.i, self.i + self.l))
            self.i += self.l


def test_preload_buffer_uniformity():
    """
    With underlying SteadyCounter and window size 200, take 1000 steps.
    Ensure 95% of values between 0 and 100 are emitted.
    """
    dataset = Preload_Buffer_Dataset(SteadyCounter(1), 200)
    loader = iter(dataset)
    outs = []

    for _ in range(1000):
        x = next(loader)[0]
        if x < 100:
            outs.append(x)

    assert len(outs) > 95, f"Only {len(outs)} values <100 detected"
