import torch
import torch.nn as nn

class GELU(nn.Module):

    def forward(self, x):

        # TODO

        raise NotImplementedError()


class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(
            # TODO
        )

    def forward(self, x):
        return self.layers(x)