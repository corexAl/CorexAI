import torch.nn as nn

from .attention import SelfAttention
from .layers import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        heads: int
    ):

        super().__init__()

        self.attention = SelfAttention(
            embedding_size,
            heads
        )

        self.feedforward = FeedForward(
            embedding_size
        )

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        x = self.norm1(
            x + self.attention(x)
        )

        x = self.norm2(
            x + self.feedforward(x)
        )

        return x
