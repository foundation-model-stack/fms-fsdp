import os
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)


class Checkpointer:
    def __init__(
        self,
        path,
    ):
        self.ckpt_path = os.path.join(path, "checkpoints")
        self.checkpoint_id = self.get_checkpoint_id()

    def load(
        self,
        model,
        optimizer,
        train_state,
    ):
        if self.checkpoint_id is None:
            self.report(
                f"No valid checkpoint detected in {self.ckpt_path}, starting from scratch..."
            )
            return model, optimizer, train_state
        self.report(f"Prior checkpoint {self.checkpoint_id} detected.")
        start_time = time.monotonic()
        state_dict = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "train_state": train_state,
        }
        dcp.load(state_dict=state_dict, checkpoint_id=self.checkpoint_id)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model_state"],
            optim_state_dict=state_dict["optim_state"],
            options=StateDictOptions(strict=False),
        )
        train_state = state_dict["train_state"]
        self.report(
            f"checkpoint loaded from {self.checkpoint_id} in {time.monotonic() - start_time:.2f} seconds."
        )
        return model, optimizer, train_state

    def save(
        self,
        step,
        model,
        optimizer,
        train_state,
    ):
        checkpoint_id = os.path.join(self.ckpt_path, f"step_{step}_ckp")

        start_time = time.monotonic()
        model_state, optim_state = get_state_dict(model, optimizer)
        state_dict = {
            "model_state": model_state,
            "optim_state": optim_state,
            "train_state": train_state,
        }
        dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_id)
        self.report(
            f"checkpoint saved in {checkpoint_id} in {time.monotonic() - start_time:.2f} seconds."
        )

    def get_checkpoint_id(self):
        if os.path.exists(self.ckpt_path) and len(os.listdir(self.ckpt_path)) > 0:
            latest_checkpoint_id = max(
                [os.path.join(self.ckpt_path, x) for x in os.listdir(self.ckpt_path)],
                key=os.path.getctime,
            )
            return latest_checkpoint_id

    def report(self, msg):
        if torch.distributed.get_rank() == 0:
            print(msg)
