"""Stage 2: bake selected images into memmapped uint8 arrays.

A Dataset that JPEG-decodes and rasterises polygons per sample starves the GPU on Windows,
where DataLoader workers are spawned and would each re-parse the annotation file. Raw uint8
memmaps remove COCO from the hot path, and the OS page cache holds the whole subset in RAM
after the first epoch.

    python -m src.data.build_cache --delete-images
    python -m src.data.build_cache --test --delete-images

`--delete-images` removes each JPEG once baked. On a nearly full disk that is what lets the
3.7 GB cache exist at all, and a re-download is resumable if it is ever needed again.

Files written to the cache directory:
    images.npy     uint8 (N, S, S, 3)
    instmaps.npy   uint8 (N, S, S)       0 = background, k = instance id
    inst_cats.npy  int16 (N, max_inst)   contiguous category of instance k at [k - 1], else -1
    labels.npy     uint8 (N, n_cats)     multi-hot, categories that own at least one patch
    meta.json, splits.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from ..config import apply_cli_overrides, load_config
from .download_subset import anns_for_image
from .grid import canonical_square, crop_resize_np, pool_instance_grid
from .raster import rasterize_canonical, sort_and_truncate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--test", action="store_true", help="bake val2017 instead of train")
    ap.add_argument("--delete-images", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    cache = Path(cfg.data.test_cache_dir if args.test else cfg.data.cache_dir)
    sel_path = cache / "selected_ids.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} missing; run src.data.download_subset first")

    sel = json.loads(sel_path.read_text("utf-8"))
    cat_remap = {int(c): i for i, c in enumerate(sel["cat_ids"])}
    root = Path(cfg.data.coco_images)

    # Drop records whose JPEG never arrived, so a partial download degrades gracefully
    # instead of writing zero-filled rows into the cache.
    records = [r for r in sel["records"] if (root / r["file_rel"]).exists()]
    if len(records) < len(sel["records"]):
        print(f"[warn] {len(sel['records']) - len(records):,} images missing on disk")
    if not records:
        raise SystemExit("no images on disk; run the downloader")

    coco = COCO(cfg.data.coco_ann_test if args.test else cfg.data.coco_ann)
    n, size, max_inst = len(records), cfg.data.cache_size, cfg.data.max_instances
    n_cats = len(sel["cat_ids"])

    images = np.lib.format.open_memmap(cache / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, size, size, 3))
    instmaps = np.lib.format.open_memmap(cache / "instmaps.npy", mode="w+", dtype=np.uint8,
                                         shape=(n, size, size))
    inst_cats = np.full((n, max_inst), -1, np.int16)
    labels = np.zeros((n, n_cats), np.uint8)
    print(f"writing cache: {n:,} x {size}x{size} "
          f"({(images.nbytes + instmaps.nbytes) / 1024**3:.1f} GB)")

    coverage = np.zeros(n, np.float32)
    for i, rec in enumerate(tqdm(records, desc="cache", unit="img")):
        path = root / rec["file_rel"]
        img = Image.open(path).convert("RGB")
        meta = coco.imgs[rec["image_id"]]
        anns = sort_and_truncate(anns_for_image(coco, rec["image_id"]), max_inst)

        # The image and the label map travel through the SAME geometry, and the label map
        # never sees a bilinear resize (rasterize_canonical scales the polygons instead).
        images[i] = np.asarray(canonical_square(img, size), dtype=np.uint8)
        instmaps[i] = rasterize_canonical(anns, int(meta["height"]), int(meta["width"]), size)

        cats = [cat_remap[int(a["category_id"])] for a in anns]
        inst_cats[i, :len(cats)] = cats
        # Labels come from the exact grid the model sees (full frame at model resolution), so
        # an image is never labelled with an object that owns no patch at evaluation time.
        grid = pool_instance_grid(
            crop_resize_np(instmaps[i], (0, 0, size, size), cfg.data.img_size, nearest=True),
            cfg.data.patch_size, max_inst, cfg.data.patch_assign_thresh)
        present = np.unique(grid[grid != 0])
        labels[i, [cats[k - 1] for k in present if k - 1 < len(cats)]] = 1
        coverage[i] = float((grid != 0).mean())

        if args.delete_images:
            path.unlink()

    images.flush()
    instmaps.flush()
    del images, instmaps
    np.save(cache / "inst_cats.npy", inst_cats)
    np.save(cache / "labels.npy", labels)

    if args.test:
        splits = {"test": list(range(n))}
    else:
        n_val = min(cfg.data.n_val, max(1, n // 10))
        splits = {"train": list(range(n - n_val)), "val": list(range(n - n_val, n))}
    (cache / "splits.json").write_text(json.dumps(splits), "utf-8")

    (cache / "meta.json").write_text(json.dumps({
        "n": n, "cache_size": size, "img_size": cfg.data.img_size,
        "patch_size": cfg.data.patch_size, "max_instances": max_inst,
        "patch_assign_thresh": cfg.data.patch_assign_thresh,
        "mask_ratio": sel["mask_ratio"], "cov_min": sel["cov_min"],
        "cat_ids": sel["cat_ids"], "cat_names": sel["cat_names"], "n_cats": n_cats,
    }, indent=1), "utf-8")

    print(f"cache built: {n:,} images, " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))
    print(f"coverage mean={coverage.mean():.3f} p10={np.percentile(coverage, 10):.3f} "
          f"p90={np.percentile(coverage, 90):.3f}")


if __name__ == "__main__":
    main()
