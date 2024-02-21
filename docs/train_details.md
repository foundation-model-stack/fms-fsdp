# Training Details

The model training used PyTorch FSDP with no activation recomputation, hybrid sharding with model
weights and optimizer state sharded within a node and data parallel across nodes, per GPU batch size of
2 (effective batch size of 1M tokens/batch), AdamW optimizer with beta1 of 0.9 and beta2 of 0.95, weight
decay of 0.1, and a learning rate ending at 3e-5 with a warmup to max learning rate of 3e-4 and a cosine
schedule to reduce to 3e-5 over 2T tokens. The loss curve tracks that of Llama2 paper and reaches a lower
loss than Llama2 7B does, which we believe is the characteristic of the dataset.

TODO: all the graphs: loss curve, etc. to be posted here

## Lesson learned

### Stability

Training was stable with no crashes, started mid-December and finished Feb 7th. We had a few hiccups as
outlined below.

**0-200B tokens:** We observed a slowdown in the iteration time (time taken to execute one training step).
We stopped the job (freeing up GPUs for other workloads) to ensure that the data loader was not causing
any slowdowns and the checkpointing was performant and accurate. We did not find any issues. By this
time, our checkpointing code had been merged into PyTorch, so we took this opportunity to make the
switch to PyTorch checkpointing code.

**200B tokens-1.9T:** We did not do any manual intervention in the job and forgot that it was running during
the winter break. When we came back early January, disk space on LustreFS had exceeded and
checkpoints were failing to be written, although training job continued. The last known checkpoint was
1.5T.

**1.5T-1.7T:** We evaluated the 1.5T checkpoint with lm-eval harness and discovered that model has been
trained with extra special token between two documents due to a formatting change that occurred
between Blue Pile V0.6 and Blue Pile 0.7. We modified the dataloader to eliminate the extra special token,
and continued training with the modified dataloader from 1.7T token onwards.

**1.7T-2T:** The loss initially spiked due to the change in the special tokens which was quickly recovered in
a few billion tokens. The training finished without any other manual intervention!!

### Speedups

There are two approaches to speeding up the performance even further. With our recent work on
improving inference speeds, we fused several layers that resulted in reduced inference latencies. We
expect these techniques to benefit training as well.

Further, with the release of a similar training code by OLMo, the issue that we had raised with PyTorch to
get compile working for FSDP increased in priority. We are currently engaged with the PT team on enabling
compile, which can provide further boost to the training speeds.
