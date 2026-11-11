"""Stage 1: curate a COCO subset from annotations alone, then fetch only what survives.

The filter runs on the annotation JSON: image sizes live there and non-crowd segmentations
are polygons we can rasterise ourselves, so no image is downloaded just to decide whether to
keep it. That keeps this project to about 3 GB of JPEGs instead of the 18 GB train2017 zip.

It also resolves the masking budget. See `choose_mask_ratio`: the ratio is not chosen by hand,
it is derived from the measured data, because the experiment is only identifiable when
instance coverage EXCEEDS the masking ratio.

    python -m src.data.download_subset              # curated train + val
    python -m src.data.download_subset --test       # all of COCO val2017, the held-out test set
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

from ..config import Config, apply_cli_overrides, load_config
from .grid import pool_instance_grid
from .raster import rasterize_canonical, sort_and_truncate

MARGIN = 0.10  # required headroom of coverage above the masking ratio
CANDIDATE_RATIOS = (0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35)


def file_rel(img_meta: dict) -> str:
    """`<split>/<id>.jpg`, taken from the image's own URL."""
    parts = img_meta["coco_url"].rsplit("/", 2)
    return f"{parts[-2]}/{parts[-1]}"


def anns_for_image(coco: COCO, image_id: int) -> list[dict]:
    """Non-crowd annotations only. A crowd region covers a group, not one object."""
    return coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False))


def analyse(img_meta: dict, anns: list[dict], cfg: Config) -> dict | None:
    """Structural statistics for one image, or None if it fails a hard criterion."""
    h, w = int(img_meta["height"]), int(img_meta["width"])
    if min(h, w) < cfg.data.min_side_px or len(anns) < cfg.data.min_instances:
        return None

    inst = rasterize_canonical(sort_and_truncate(anns, cfg.data.max_instances),
                               h, w, cfg.data.img_size)
    grid = pool_instance_grid(inst, cfg.data.patch_size, cfg.data.max_instances,
                              cfg.data.patch_assign_thresh)

    flat = grid.reshape(-1)
    ids, counts = np.unique(flat[flat != 0], return_counts=True)
    keep = counts >= cfg.data.min_instance_patches
    ids, counts = ids[keep], counts[keep]
    if len(ids) < cfg.data.min_instances:
        return None

    cov = float(counts.sum() / grid.size)
    largest = float(counts.max() / grid.size)
    if largest > cfg.data.largest_instance_max or cov > cfg.data.cov_max:
        return None

    return {"image_id": int(img_meta["id"]), "n_inst": int(len(ids)),
            "coverage": round(cov, 4), "largest": round(largest, 4),
            "coco_url": img_meta["coco_url"], "file_rel": file_rel(img_meta)}


def ratio_table(coverages: np.ndarray, largest: np.ndarray, margin: float = MARGIN,
                candidates=CANDIDATE_RATIOS) -> list[tuple[float, float, int, int]]:
    """(r, cov_min, n_coverage_only, n_joint) for every candidate ratio."""
    out = []
    for r in candidates:
        cov_ok = coverages >= r + margin
        out.append((r, r + margin, int(cov_ok.sum()), int((cov_ok & (largest <= r)).sum())))
    return out


def choose_mask_ratio(coverages: np.ndarray, largest: np.ndarray, n_needed: int,
                      margin: float = MARGIN, candidates=CANDIDATE_RATIOS
                      ) -> tuple[float | None, float, int]:
    """Pick the largest masking ratio r satisfying BOTH constraints for `n_needed` images:

        coverage >= r + margin      instances must OVERFLOW the budget
        largest  <= r               every instance must be able to FIT the budget

    The first is the identifiability condition. If instances cover less than the budget,
    every instance must be masked to fill it, so uniform and adaptive weighting select the
    same set and arms B and C become indistinguishable.

    The second exists because skip-fit never masks an instance that does not fit the
    remaining budget, so an instance larger than M can NEVER be masked. Measured on COCO at
    r = 0.45 without this constraint: only 8% of instances were oversized, but they appeared
    in 45% of images and held 76% of the instance patches there, so the main subject was
    never removed from nearly half the dataset.

    The two pull in opposite directions, so r is solved for rather than picked.
    """
    for r, cov_min, _, n_joint in ratio_table(coverages, largest, margin, candidates):
        if n_joint >= n_needed:
            return r, cov_min, n_joint
    return None, 0.0, 0


def _fetch(session: requests.Session, url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=(5, 30)) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 16):
                        fh.write(chunk)
            os.replace(tmp, dest)  # atomic: Ctrl-C never leaves a truncated JPEG
            return True
        except Exception:
            time.sleep(2**attempt)
    tmp.unlink(missing_ok=True)
    return False


def download_images(records: list[dict], root: Path, workers: int) -> int:
    """Fetch every record to `root / file_rel`. Resumable: files on disk are skipped."""
    sessions = [requests.Session() for _ in range(workers)]

    def job(item):
        i, rec = item
        dest = root / rec["file_rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        return _fetch(sessions[i % workers], rec["coco_url"], dest)

    failures = 0
    with ThreadPoolExecutor(workers) as pool:
        for ok in tqdm(pool.map(job, enumerate(records)), total=len(records),
                       desc="download", unit="img"):
            failures += int(not ok)
    return failures


def curate_train(cfg: Config, limit: int = 0) -> list[dict]:
    """Scan train2017, resolve the masking ratio, write `selected_ids.json`."""
    cache = Path(cfg.data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    coco = COCO(cfg.data.coco_ann)
    cat_ids = sorted(coco.getCatIds())
    img_ids = sorted(coco.imgs)[:limit or None]

    records = []
    for image_id in tqdm(img_ids, desc="filter", unit="img"):
        rec = analyse(coco.imgs[image_id], anns_for_image(coco, image_id), cfg)
        if rec is not None:
            records.append(rec)
    if not records:
        raise SystemExit("no images passed the structural filter; check the config")

    coverages = np.array([r["coverage"] for r in records])
    largest = np.array([r["largest"] for r in records])
    n_needed = cfg.data.n_train + cfg.data.n_val

    print("\n      r    cov_min   n(coverage)   n(coverage & largest<=r)")
    table = ratio_table(coverages, largest)
    for r_, cmin_, n_cov, n_joint in table:
        flag = "  <-- feasible" if n_joint >= n_needed else ""
        print(f"   {r_:.2f}    {cmin_:.2f}      {n_cov:8,}      {n_joint:8,}{flag}")

    ratio, cov_min, n_avail = choose_mask_ratio(coverages, largest, n_needed)
    if ratio is None:
        best = max(table, key=lambda t: t[3])
        raise SystemExit(f"GATE FAILED: no ratio satisfies both constraints for {n_needed:,} "
                         f"images. Best is r={best[0]:.2f} with {best[3]:,}. Lower "
                         "data.n_train rather than relaxing the constraints.")

    selected = [r for r in records if r["coverage"] >= cov_min and r["largest"] <= ratio]
    np.random.default_rng(cfg.data.subset_seed).shuffle(selected)
    selected = selected[:n_needed]

    report = {
        "n_scanned": len(img_ids),
        "n_passed_structure": len(records),
        "resolved_mask_ratio": ratio,
        "resolved_cov_min": round(cov_min, 4),
        "margin": MARGIN,
        "n_available": n_avail,
        "n_selected": len(selected),
        "ratio_table": [{"r": r_, "cov_min": round(c_, 3), "n_coverage": nc, "n_joint": nj}
                        for r_, c_, nc, nj in table],
        "selected_coverage_mean": round(float(np.mean([r["coverage"] for r in selected])), 4),
        "selected_largest_mean": round(float(np.mean([r["largest"] for r in selected])), 4),
        "selected_n_inst_mean": round(float(np.mean([r["n_inst"] for r in selected])), 2),
        "coverage_hist": {
            "bins": np.round(np.linspace(0, 1, 21), 3).tolist(),
            "counts": np.histogram(coverages, bins=np.linspace(0, 1, 21))[0].tolist(),
        },
    }
    (cache / "filter_report.json").write_text(json.dumps(report, indent=2), "utf-8")
    (cache / "selected_ids.json").write_text(json.dumps({
        "mask_ratio": ratio, "cov_min": cov_min, "cat_ids": cat_ids,
        "cat_names": [coco.cats[c]["name"] for c in cat_ids], "records": selected}), "utf-8")

    print(f"\nresolved masking ratio r = {ratio:.2f} (M = "
          f"{int(round(ratio * cfg.num_patches))} of {cfg.num_patches} patches)")
    print(f"{n_avail:,} images qualify, {len(selected):,} selected; mean coverage "
          f"{report['selected_coverage_mean']:.3f}, mean instances "
          f"{report['selected_n_inst_mean']:.1f}")
    return selected


def curate_test(cfg: Config) -> list[dict]:
    """Every val2017 image with at least one object. No curation on purpose: the test set
    should say how the features behave on ordinary COCO, not on the cluttered training
    subset, and these images are never seen during pretraining or head selection."""
    cache = Path(cfg.data.test_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    coco = COCO(cfg.data.coco_ann_test)
    cat_ids = sorted(coco.getCatIds())
    records = [{"image_id": int(i), "coco_url": coco.imgs[i]["coco_url"],
                "file_rel": file_rel(coco.imgs[i])}
               for i in sorted(coco.imgs) if anns_for_image(coco, i)]
    (cache / "selected_ids.json").write_text(json.dumps({
        "mask_ratio": None, "cov_min": None, "cat_ids": cat_ids,
        "cat_names": [coco.cats[c]["name"] for c in cat_ids], "records": records}), "utf-8")
    print(f"test set: {len(records):,} of {len(coco.imgs):,} val2017 images have objects")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--test", action="store_true", help="prepare val2017 instead of train")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: scan only N images")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    ann = Path(cfg.data.coco_ann_test if args.test else cfg.data.coco_ann)
    if not ann.exists():
        raise SystemExit(f"annotations not found: {ann}\nSee the README, 'Reproducing'.")

    selected = curate_test(cfg) if args.test else curate_train(cfg, args.limit)
    if args.skip_download:
        return
    failures = download_images(selected, Path(cfg.data.coco_images), args.workers)
    print(f"done, {failures} failures (re-run to retry; the fetch is resumable)")


if __name__ == "__main__":
    main()
