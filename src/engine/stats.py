"""Metrics and the statistics behind every comparison in the results.

Macro mAP uses a FIXED class set. Choosing the class set per sample lets rare classes fall out
of some bootstrap replicates, which inflates those replicates and can push a point estimate
outside its own interval.

Comparisons are between arms on the same images and the same seeds, so they are paired, and
they resample BOTH sources of noise: which seeds were run and which test images were drawn.
A bootstrap over images alone is conditional on one training run and is far too confident
about a difference between two networks.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# With k positives, a size-N resample contains none with probability about exp(-k), so k >= 10
# makes it negligible. Rarer classes are left out of the macro average entirely.
MIN_POSITIVES = 10


def valid_classes(y_true: np.ndarray, min_pos: int = MIN_POSITIVES) -> np.ndarray:
    pos = y_true.sum(0)
    return (pos >= min_pos) & (pos <= len(y_true) - min_pos)


def ap_columns(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Average precision for every column at once.

    Same definition as sklearn's binary case, AP = (1 / n_pos) * sum over positive ranks of
    precision@rank, computed in a few vectorised ops because the bootstrap needs it millions
    of times. `tests/test_stats.py` pins agreement with sklearn. A class with no positives
    scores 0.0 rather than being dropped, so every replicate averages the same denominator.
    """
    order = np.argsort(-y_score, axis=0, kind="stable")
    yt = np.take_along_axis(y_true, order, axis=0).astype(np.float64)
    tp = np.cumsum(yt, axis=0)
    ranks = np.arange(1, yt.shape[0] + 1, dtype=np.float64)[:, None]
    n_pos = yt.sum(0)
    return np.where(n_pos > 0, ((tp / ranks) * yt).sum(0) / np.maximum(n_pos, 1.0), 0.0)


def macro_ap(y_true: np.ndarray, y_score: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Mean AP over a fixed set of classes."""
    valid = valid_classes(y_true) if valid is None else valid
    cols = np.flatnonzero(valid)
    if cols.size == 0:
        return float("nan")
    return float(ap_columns(y_true[:, cols], y_score[:, cols]).mean())


def iou_counts(pred: np.ndarray, target: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-image (N, 3, C) counts of [intersection, predicted, target] over the patch grid.

    Keeping counts per image is what lets the bootstrap resample images for the dense head.
    """
    n = pred.shape[0]
    rows = np.arange(n)[:, None]
    out = np.zeros((n, 3, n_classes), np.int32)
    np.add.at(out[:, 1, :], (rows, pred), 1)
    np.add.at(out[:, 2, :], (rows, target), 1)
    hit = pred == target
    np.add.at(out[:, 0, :], (rows.repeat(pred.shape[1], 1)[hit], pred[hit]), 1)
    return out


def mean_iou(counts: np.ndarray, valid: np.ndarray) -> float:
    """Mean IoU over a fixed class set, from `iou_counts` summed over the sampled images."""
    inter, pred, target = counts.sum(0).astype(np.float64)
    union = pred + target - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    return float(iou[valid].mean())


def seed_image_bootstrap(metric: Callable[[object, np.ndarray], float], runs: dict[str, list],
                         n_images: int, contrasts: list[tuple[str, str]], n_boot: int,
                         seed: int = 0) -> dict[tuple[str, str], np.ndarray]:
    """Bootstrap distribution of mean paired differences between arms.

    `runs[arm]` is a list with one entry per seed, in the same seed order for every arm, and
    `metric(run, idx)` scores one run on the images `idx`. Each replicate resamples the images
    (shared by all arms and seeds, because they really are the same images) and then the seeds
    (shared by both arms of a contrast, because a seed fixes the init and data order of both).

    Returns {contrast: (n_boot,) differences}.
    """
    rng = np.random.default_rng(seed)
    arms = sorted({a for c in contrasts for a in c})
    n_seeds = len(runs[arms[0]])
    out = {c: np.zeros(n_boot) for c in contrasts}
    for b in range(n_boot):
        idx = rng.integers(0, n_images, n_images)
        scores = {a: np.array([metric(r, idx) for r in runs[a]]) for a in arms}
        picked = rng.integers(0, n_seeds, n_seeds)
        for a, c in contrasts:
            out[(a, c)][b] = (scores[a][picked] - scores[c][picked]).mean()
    return out


def summarise_bootstrap(diffs: np.ndarray) -> tuple[float, float, float]:
    """(ci_low, ci_high, two-sided p). p is twice the smaller tail mass, floored at 2 / B."""
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(lo), float(hi), float(min(max(p, 2.0 / len(diffs)), 1.0))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values for one family of tests."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
