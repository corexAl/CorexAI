import torch
import torch.nn as nn

from .transformer import TransformerBlock


class CorexModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        model_cfg = config["model"]

        self.embedding = nn.Embedding(
            model_cfg["vocab_size"],
            model_cfg["embedding_size"]
        )

        self.position_embedding = nn.Embedding(
            model_cfg["context_length"],
            model_cfg["embedding_size"]
        )

        self.layers = nn.ModuleList(

            TransformerBlock(
                model_cfg["embedding_size"],
                model_cfg["attention_heads"]
            )

            for _ in range(
                model_cfg["layers"]
            )

        )

        self.output = nn.Linear(
            model_cfg["embedding_size"],
            model_cfg["vocab_size"]
        )

    def forward(self, tokens):

        batch, sequence = tokens.shape

        positions = torch.arange(
            sequence,
            device=tokens.device
        ).unsqueeze(0)

        x = (
            self.embedding(tokens)
            + self.position_embedding(positions)
        )

        for layer in self.layers:

            x = layer(x)

        return self.output(x)
