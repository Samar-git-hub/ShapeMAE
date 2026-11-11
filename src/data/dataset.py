"""Cached COCO instance dataset.

Two decisions are load-bearing for throughput on Windows:

  1. Samples are returned as uint8 HWC and normalised on the GPU. float32 CHW would push four
     times the bytes through the spawn-based DataLoader pipes.
  2. Memmaps are opened lazily and dropped before pickling. Pickling an np.memmap serialises
     its whole buffer, and Windows workers receive the Dataset by pickle.

Augmentation is RandomResizedCrop + horizontal flip, applied to the image and the instance
map through the SAME crop box, with NEAREST resampling on the label map. Pooling to the patch
grid happens after the crop, so alignment is exact by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..utils.seed import worker_init_fn
from .grid import crop_resize_np, pool_instance_grid, rrc_params

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CachedCocoInstances(Dataset):
    """Per sample:
        image     uint8  (H, W, 3)      HWC; normalise on device
        grid      int64  (L,)           instance id per patch, 0 = background
        cat_grid  int64  (L,)           category + 1 per patch, 0 = background
        inst_cats int64  (max_inst,)    contiguous category per instance id, -1 = absent
        labels    float32(n_cats,)      multi-hot, for the classification heads
    """

    def __init__(self, cfg: Config, mode: str = "train", augment: bool | None = None) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be train|val|test, got {mode!r}")
        self.cfg = cfg
        self.mode = mode
        # Separate from `mode`: which split to read and whether to crop randomly are different
        # questions. Feature extraction reads the train split with fixed geometry, otherwise
        # the features, and so every head's score, would change on each run.
        self.augment = (mode == "train") if augment is None else bool(augment)
        self.cache = Path(cfg.data.test_cache_dir if mode == "test" else cfg.data.cache_dir)

        meta_path = self.cache / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing; see the README, 'Reproducing'")
        meta = json.loads(meta_path.read_text("utf-8"))
        self.indices: list[int] = json.loads((self.cache / "splits.json").read_text("utf-8"))[mode]
        self.cat_names: list[str] = meta["cat_names"]
        self.n_cats = int(meta["n_cats"])
        self.cache_size = int(meta["cache_size"])
        self.max_inst = int(meta["max_instances"])

        self._images: np.ndarray | None = None
        self._inst: np.ndarray | None = None
        self._inst_cats: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in ("_images", "_inst", "_inst_cats", "_labels", "_rng"):
            state[key] = None  # each worker reopens on first use
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def _augment_rng(self) -> np.random.Generator:
        """torch.initial_seed() inside a worker is base_seed + worker_id, derived from the
        global seed: each worker gets a distinct, reproducible stream."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _ensure_open(self) -> None:
        if self._images is None:
            self._images = np.load(self.cache / "images.npy", mmap_mode="r")
            self._inst = np.load(self.cache / "instmaps.npy", mmap_mode="r")
            self._inst_cats = np.load(self.cache / "inst_cats.npy", mmap_mode="r")
            self._labels = np.load(self.cache / "labels.npy", mmap_mode="r")

    def __getitem__(self, i: int):
        self._ensure_open()
        idx = self.indices[i]
        cfg = self.cfg
        out, src = cfg.data.img_size, self.cache_size

        img = np.asarray(self._images[idx])
        inst = np.asarray(self._inst[idx])

        flip = False
        box = (0, 0, src, src)  # full frame: same field of view as the curation statistic
        if self.augment:
            rng = self._augment_rng()
            box = rrc_params(src, src, out, cfg.data.rrc_scale, cfg.data.rrc_ratio, rng)
            flip = cfg.data.hflip and bool(rng.integers(0, 2))

        img = crop_resize_np(img, box, out, nearest=False)
        inst = crop_resize_np(inst, box, out, nearest=True)
        if flip:
            img, inst = img[:, ::-1], inst[:, ::-1]

        grid = pool_instance_grid(inst, cfg.data.patch_size, self.max_inst,
                                  cfg.data.patch_assign_thresh)
        inst_cats = np.asarray(self._inst_cats[idx], np.int64)

        flat = grid.reshape(-1).astype(np.int64)
        cat_grid = np.zeros_like(flat)
        obj = flat > 0
        if obj.any():
            resolved = inst_cats[flat[obj] - 1]  # the only correct lookup; see grid.py
            cat_grid[obj] = np.where(resolved >= 0, resolved + 1, 0)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "grid": torch.from_numpy(flat),
            "cat_grid": torch.from_numpy(cat_grid),
            "inst_cats": torch.from_numpy(inst_cats),
            "labels": torch.from_numpy(np.asarray(self._labels[idx], np.float32)),
        }


def normalize_batch(images_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> normalised float32 (N, 3, H, W) on device, so the host-to-device
    copy stays uint8."""
    x = images_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `normalize_batch`, clamped to [0, 1] for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp_(0, 1)


def build_loader(cfg: Config, mode: str, device: torch.device, shuffle: bool | None = None,
                 batch_size: int | None = None, augment: bool | None = None,
                 workers: int | None = None):
    """`augment=None` keeps the default: on for the train split, off otherwise."""
    ds = CachedCocoInstances(cfg, mode, augment=augment)
    nw = cfg.data.num_workers if workers is None else min(cfg.data.num_workers, workers)
    nw = nw if device.type == "cuda" else 0
    return ds, DataLoader(
        ds,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=(mode == "train") if shuffle is None else shuffle,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        drop_last=(mode == "train" and shuffle is not False),
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
        worker_init_fn=worker_init_fn if nw > 0 else None,
    )
