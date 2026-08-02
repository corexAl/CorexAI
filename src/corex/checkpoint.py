from pathlib import Path

import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)

        self.directory.mkdir(
            exist_ok=True
        )


    def save(
        self,
        model,
        optimizer,
        epoch
    ):

        torch.save(

            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            },

            self.directory /
            f"epoch_{epoch}.pt"

        )


    def load(
        self,
        model,
        optimizer,
        path
    ):

        checkpoint = torch.load(path)

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        return checkpoint["epoch"]
