"""Generate a synthetic cache with the exact on-disk layout of the real one.

Lets the whole pipeline (training, evaluation, heads, analysis, figures) run end to end
without COCO. The images mean nothing; shapes, dtypes and index conventions match production.

    python -m src.data.synthetic --out data/smoke --n 192
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .grid import crop_resize_np, pool_instance_grid


def build(out: Path, n: int, size: int, img_size: int, patch: int, n_cats: int,
          max_instances: int, ratio: float, seed: int, splits: dict[str, list[int]] | None = None
          ) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, size, size, 3))
    instmaps = np.lib.format.open_memmap(out / "instmaps.npy", mode="w+", dtype=np.uint8,
                                         shape=(n, size, size))
    inst_cats = np.full((n, max_instances), -1, np.int16)
    labels = np.zeros((n, n_cats), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]

    for i in range(n):
        # Smooth-ish background so per-patch normalisation has non-degenerate variance.
        base = rng.integers(40, 216, (8, 8, 3)).astype(np.float32)
        big = np.repeat(np.repeat(base, -(-size // 8), 0), -(-size // 8), 1)[:size, :size]
        img = np.clip(big + rng.normal(0, 12, (size, size, 3)), 0, 255)

        inst = np.zeros((size, size), np.uint8)
        n_inst = int(rng.integers(3, max_instances // 2 + 3))
        cats = rng.integers(0, n_cats, n_inst)
        for k in range(1, n_inst + 1):  # bright ellipse per object, colour keyed to category
            r = int(size * rng.uniform(0.10, 0.25))
            cy, cx = int(rng.integers(0, size)), int(rng.integers(0, size))
            ell = ((yy - cy) / r) ** 2 + ((xx - cx) / (r * 0.8)) ** 2 <= 1
            inst[ell] = k
            img[ell] = 40 + 200 * (cats[k - 1] + 1) / n_cats

        images[i] = img.astype(np.uint8)
        instmaps[i] = inst
        inst_cats[i, :n_inst] = cats
        grid = pool_instance_grid(crop_resize_np(inst, (0, 0, size, size), img_size, True),
                                  patch, max_instances, 0.25)
        labels[i, [cats[k - 1] for k in np.unique(grid[grid != 0])]] = 1

    images.flush()
    instmaps.flush()
    del images, instmaps
    np.save(out / "inst_cats.npy", inst_cats)
    np.save(out / "labels.npy", labels)

    if splits is None:
        n_val = max(2, n // 5)
        splits = {"train": list(range(n - n_val)), "val": list(range(n - n_val, n))}
    (out / "splits.json").write_text(json.dumps(splits), "utf-8")
    (out / "meta.json").write_text(json.dumps({
        "n": n, "cache_size": size, "img_size": img_size, "patch_size": patch,
        "max_instances": max_instances, "patch_assign_thresh": 0.25, "mask_ratio": ratio,
        "cov_min": ratio + 0.1, "cat_ids": list(range(1, n_cats + 1)),
        "cat_names": [f"synth_{i:02d}" for i in range(n_cats)], "n_cats": n_cats,
    }), "utf-8")
    print(f"synthetic cache -> {out}: {n} images, {size}px cache, {img_size}px model input")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/smoke")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--n-cats", type=int, default=6)
    ap.add_argument("--max-instances", type=int, default=12)
    ap.add_argument("--ratio", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()

    out = Path(args.out)
    build(out / "train", args.n, args.size, args.img_size, args.patch, args.n_cats,
          args.max_instances, args.ratio, args.seed)
    n_test = max(16, args.n // 4)
    build(out / "test", n_test, args.size, args.img_size, args.patch, args.n_cats,
          args.max_instances, args.ratio, args.seed + 1, {"test": list(range(n_test))})


if __name__ == "__main__":
    main()
