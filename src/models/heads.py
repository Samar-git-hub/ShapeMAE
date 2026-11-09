"""Heads that read a pretrained encoder. Each one asks a different question of the features.

    linear       Can the classes be read off the mean-pooled features with a hyperplane?
    class_attn   Is the information in the patches, but blurred by mean pooling?
    dense        Is the information spatially localised, patch by patch?

`ClassAttentionHead` is the linear head plus one idea: every class learns where to look. With
uniform attention it reduces exactly to a linear classifier on the mean-pooled tokens, and that
is where it starts, so any gain over `linear` comes purely from the learned pooling, and its
attention map is a direct picture of what each class reads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassAttentionHead(nn.Module):
    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.where = nn.Linear(dim, n_classes)  # per class, how much each patch matters
        self.what = nn.Linear(dim, n_classes)   # per class, the evidence each patch gives
        nn.init.zeros_(self.where.weight)       # attention starts uniform
        nn.init.zeros_(self.where.bias)

    def attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, L, D) -> (N, L, C), a distribution over patches for every class."""
        return self.where(tokens).softmax(dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return (self.attention(tokens) * self.what(tokens)).sum(dim=1)


class DenseHead(nn.Module):
    """One linear classifier applied to every patch token, class 0 being background."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, n_classes + 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc(tokens)
