"""Turn finished runs into the tables and contrasts reported in the README.

    python -m src.engine.analyze --split val     # means only; never touches the test set
    python -m src.engine.analyze --split test    # means + the pre-planned contrasts

Everything here reads files written by the other stages, so it needs no GPU.

The contrasts were fixed before any test result existed. Each moves exactly one variable:

    shuffled - random     geometry   scattered holes -> object-sized blobs, no alignment
    instance - shuffled   semantics  someone else's objects -> this image's objects
    adaptive - instance   adaptivity uniform -> difficulty-weighted categories

They form one family of tests across all heads, so p-values are Holm-adjusted together.
`instance - random` is reported as well, as the total effect, but it claims nothing the other
contrasts do not and sits outside the family.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, apply_cli_overrides, load_config
from .stats import (
    holm,
    macro_ap,
    mean_iou,
    seed_image_bootstrap,
    summarise_bootstrap,
    valid_classes,
)

ARMS = ("random", "shuffled", "instance", "adaptive")
HEADS = ("knn_raw", "knn_std", "linear", "class_attn", "dense_miou", "finetune")
FAMILY = (("shuffled", "random"), ("instance", "shuffled"), ("adaptive", "instance"))
EXTRA = (("instance", "random"),)


def seeds_of(runs: Path, arm: str) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in (runs / arm).glob("seed_*"))


def load_heads(runs: Path, split: str) -> pd.DataFrame:
    """Long table: one row per (arm, seed, head)."""
    rows = []
    for arm in ARMS:
        for seed in seeds_of(runs, arm):
            path = runs / arm / f"seed_{seed}" / f"heads_{split}.json"
            if path.exists():
                for head, value in json.loads(path.read_text("utf-8")).items():
                    rows.append({"arm": arm, "seed": seed, "head": head, "value": value})
    return pd.DataFrame(rows)


def mean_std_table(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["arm", "head"])["value"].agg(["mean", "std", "count"]).reset_index()
    order = {a: i for i, a in enumerate(ARMS)}
    return g.sort_values(["head", "arm"], key=lambda s: s.map(order) if s.name == "arm" else s)


def _load_scores(runs: Path, arm: str, seed: int, head: str) -> np.ndarray:
    d = runs / arm / f"seed_{seed}"
    if head == "finetune":
        return np.load(d / "scores_finetune.npz")["scores"].astype(np.float32)
    key = "dense_counts" if head == "dense_miou" else head
    data = np.load(d / "scores_test.npz")[key]
    return data if head == "dense_miou" else data.astype(np.float32)


def _bootstrap_head(args: tuple) -> list[dict]:
    """All contrasts for one head. Runs in its own process; heads are independent."""
    runs, head, labels, dense_valid, seeds, n_boot = args
    contrasts = list(FAMILY + EXTRA)
    scores = {a: [_load_scores(runs, a, s, head) for s in seeds] for a in ARMS}

    if head == "dense_miou":
        def metric(run, idx):
            return mean_iou(run[idx], dense_valid)
    else:
        valid = valid_classes(labels)

        def metric(run, idx):
            return macro_ap(labels[idx], run[idx], valid)

    n_images = len(labels)
    everyone = np.arange(n_images)
    point = {a: np.array([metric(r, everyone) for r in scores[a]]) for a in ARMS}
    boots = seed_image_bootstrap(metric, scores, n_images, contrasts, n_boot)

    rows = []
    for a, b in contrasts:
        delta = point[a] - point[b]
        lo, hi, p = summarise_bootstrap(boots[(a, b)])
        rows.append({"head": head, "a": a, "b": b, "diff": delta.mean(),
                     "diff_std": delta.std(ddof=1), "seeds_positive": int((delta > 0).sum()),
                     "n_seeds": len(seeds), "ci_lo": lo, "ci_hi": hi, "p": p})
    print(f"  {head}: done", flush=True)
    return rows


def contrasts(cfg: Config, runs: Path, out: Path) -> pd.DataFrame:
    """Hierarchical-bootstrap contrasts on the test set, Holm-adjusted across the family."""
    labels = np.load(Path(cfg.data.test_cache_dir) / "labels.npy").astype(np.float32)
    seeds = sorted(set.intersection(*[set(seeds_of(runs, a)) for a in ARMS]))
    counts = _load_scores(runs, ARMS[0], seeds[0], "dense_miou").sum((0,))
    dense_valid = counts[2] >= 50  # same rule as heads.py

    jobs = [(runs, h, labels, dense_valid, seeds, cfg.eval.bootstrap) for h in HEADS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
        rows = [r for part in pool.map(_bootstrap_head, jobs) for r in part]

    df = pd.DataFrame(rows)
    family = df.apply(lambda r: (r["a"], r["b"]) in FAMILY, axis=1)
    df.loc[family, "p_holm"] = holm(df.loc[family, "p"].tolist())
    df.to_csv(out / "contrasts.csv", index=False)
    return df


def diagnostics(cfg: Config, runs: Path, out: Path) -> None:
    """Reconstruction, attention distance and the adaptive policy, aggregated over seeds."""
    rec, attn = [], []
    for arm in ARMS:
        for seed in seeds_of(runs, arm):
            path = runs / arm / f"seed_{seed}" / "eval.json"
            if not path.exists():
                continue
            e = json.loads(path.read_text("utf-8"))
            rec.append({"arm": arm, "seed": seed,
                        **{k: v for k, v in e.items() if not isinstance(v, list)}})
            attn.append({"arm": arm, "seed": seed, "attn_full": np.mean(e["attn_dist_full"]),
                         "attn_masked": np.mean(e["attn_dist_masked"])})
    if rec:
        pd.DataFrame(rec).groupby("arm").agg(["mean", "std"]).drop(columns="seed").to_csv(
            out / "reconstruction.csv")
        pd.DataFrame(attn).groupby("arm").agg(["mean", "std"]).drop(columns="seed").to_csv(
            out / "attention_distance.csv")

    hist = [np.load(p) for p in sorted((runs / "adaptive").glob("seed_*/cat_ema.npy"))]
    meta_path = Path(cfg.data.cache_dir) / "meta.json"
    if hist and meta_path.exists():
        names = json.loads(meta_path.read_text("utf-8"))["cat_names"]
        final = np.stack([h[-1, 1, :] for h in hist])  # (seeds, cats) final sampling weights
        df = pd.DataFrame({"category": names, "weight_mean": final.mean(0),
                           "weight_std": final.std(0, ddof=1) if len(hist) > 1 else 0.0})
        df.sort_values("weight_mean", ascending=False).to_csv(out / "adaptive_policy.csv",
                                                              index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--results", default="results")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    runs, out = Path(cfg.train.out_dir), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    long = load_heads(runs, args.split)
    long.to_csv(out / f"heads_{args.split}_per_seed.csv", index=False)
    table = mean_std_table(long)
    table.to_csv(out / f"heads_{args.split}.csv", index=False)
    wide = long.pivot_table(index="arm", columns="head", values="value", aggfunc="mean")
    print(wide.reindex([a for a in ARMS if a in wide.index]).to_string(
        float_format=lambda x: f"{x:.4f}"))

    diagnostics(cfg, runs, out)
    if args.split == "test":
        df = contrasts(cfg, runs, out)
        print(df.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
