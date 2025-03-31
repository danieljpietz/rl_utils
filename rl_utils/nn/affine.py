from torch import nn


class Affine(nn.Module):
    def __init__(self, m: float = 1.0, b: float = 0.0):
        super().__init__()
        self.m = m
        self.b = b

    def forward(self, x):
        return self.m * x + self.b

    def __repr__(self):
        return f"Affine(m={self.m}, b={self.b})"
