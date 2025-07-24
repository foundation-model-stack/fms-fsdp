# Data Loader

We design a data loader as part of our pretraining that can provide shuffling in realtime while ensuring that there is no drop in GPU utilization. We design it to be scalable to multiple nodes (tested to 128 nodes), streaming, rescaling the number of GPUs during a single training run, and allows for restart from a given state.

## Details

The current distributed dataloader is designed to meet two important needs of data scientists running large-scale training workloads: seamless resumption of an interrupted job, and rapid iteration on dataset composition and handling.

We address the first by maintaining a checkpointable state that allows the user to restart model training from checkpoint, mid-epoch, while keeping a guarantee that each document will be viewed exactly once in any given epoch, up to oversampling (i.e. no revisiting stale data). The user is also free to scale the job up or down to different numbers of gpus from phase to phase, while still maintaining this guarantee.

To address the second concern, we enforce a rigid format on our input data (tokenized documents, in arrow shard files, organized into dataset subdirectories, with a single unified metadata file of document counts per shard) but construct the specific dataset combinations and mixes dynamically at runtime, based on user inputs. This is accomplished separately on each worker process, with no communication needed between devices. Each worker then streams through the ordered documents and shard files according to its constructed plan, pulling files from disk or cloud on demand as training proceeds. This allows the user to add or eliminate subdatasets, adjust subdataset sampling rates, change BOS/EOS/SEP tokens, toggle padding or packing on or off, adjust sequence lengths, or swap out the training task, for example, without having to build any new training datasets on disk from run to run (a potentially long and expensive process for Terabyte-scale data). 

Because each worker is streaming documents and files sequentially, shuffling is required, and this is accomplished via an internal buffer which ensures that in expectation, two consecutive lines in the stream will appear 10,000 steps apart (this can be adjusted higher or lower as desired). Finally, the dataloader is implemented as modular extensions of PyTorch Datasets, allowing the user to add or remove custom data pipeline functionality as needed.

Further technical details can be found in the `fms-fsdp/utils/dataset_utils.py` file.
