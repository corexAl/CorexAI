import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, embedding_size: int):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            )

        )

    def forward(self, x):

        return self.network(x)
