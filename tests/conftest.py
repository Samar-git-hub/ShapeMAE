"""Synthetic fixtures so the whole suite runs without COCO on disk."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data.synthetic import build  # noqa: E402

N_CATS = 8
MAX_INST = 6
CACHE_SIZE = 96
IMG_SIZE = 64
PATCH = 16  # -> 4x4 grid, L = 16


@pytest.fixture(scope="session")
def synth_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    build(root / "train", 24, CACHE_SIZE, IMG_SIZE, PATCH, N_CATS, MAX_INST, 0.5, 0,
          {"train": list(range(18)), "val": list(range(18, 24))})
    build(root / "test", 8, CACHE_SIZE, IMG_SIZE, PATCH, N_CATS, MAX_INST, 0.5, 1,
          {"test": list(range(8))})

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    cfg.data.cache_dir = str(root / "train")
    cfg.data.test_cache_dir = str(root / "test")
    cfg.data.cache_size = CACHE_SIZE
    cfg.data.img_size = IMG_SIZE
    cfg.data.patch_size = PATCH
    cfg.data.max_instances = MAX_INST
    cfg.data.num_workers = 0
    cfg.mask.ratio = 0.5
    cfg.train.batch_size = 4
    return cfg


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
