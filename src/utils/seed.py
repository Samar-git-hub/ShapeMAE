"""Seeding and per-step RNG derivation.

Bitwise determinism would need deterministic kernels and is not required: within a seed, all
arms share the initialisation, the data order and the augmentation stream, which is what a
controlled comparison needs. Variation between runs is measured directly with several seeds.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def step_rng(seed: int, epoch: int, step: int) -> np.random.Generator:
    """Deterministic per-(epoch, step) generator, independent of worker scheduling."""
    return np.random.default_rng([seed, epoch, step])


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % 2**31
    np.random.seed((base + worker_id) % 2**31)
    random.seed(base + worker_id)
