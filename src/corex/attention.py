import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        heads: int
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):

        output, _ = self.attention(
            x,
            x,
            x
        )

        return output
