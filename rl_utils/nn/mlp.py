from torch import nn
from typing import Iterable

from .affine import Affine


class MLP(nn.Module):
    def __init__(
        self,
        topology: Iterable[int],
        activation: nn.Module = nn.ReLU(),
        normalize: float | None = None,
    ):
        super().__init__()

        layers = []
        for in_dim, out_dim in zip(topology[:-1], topology[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
        layers.append(nn.Linear(topology[-2], topology[-1]))
        if normalize is not None:
            layers.append(nn.Tanh())
            layers.append(Affine(normalize))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
