"""The statistics behind every comparison.

`ap_columns` must agree with sklearn, which it replaced purely for speed, and the bootstrap
must find real differences, find none between identical arms, and widen with seed noise.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.engine.stats import (
    ap_columns,
    holm,
    iou_counts,
    mean_iou,
    seed_image_bootstrap,
    summarise_bootstrap,
    valid_classes,
)


def synth(n=400, c=12, seed=0):
    """Scores correlated with labels, so AP is meaningfully above chance."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, c)) < rng.uniform(0.08, 0.5, c)).astype(np.float64)
    s = y * rng.normal(1.0, 1.0, (n, c)) + rng.normal(0.0, 1.0, (n, c))
    return y, s


def test_ap_columns_matches_sklearn():
    y, s = synth()
    theirs = np.array([average_precision_score(y[:, c], s[:, c]) for c in range(y.shape[1])])
    assert np.allclose(ap_columns(y, s), theirs, atol=1e-9)


def test_ap_columns_handles_no_positives():
    y, s = synth()
    y[:, 0] = 0.0
    assert ap_columns(y, s)[0] == 0.0  # scored 0, not dropped


def test_ap_perfect_and_worst_ranking():
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    assert ap_columns(y, np.array([[4.0], [3.0], [2.0], [1.0]]))[0] == pytest.approx(1.0)
    assert ap_columns(y, np.array([[1.0], [2.0], [3.0], [4.0]]))[0] == pytest.approx(
        (1 / 3 + 2 / 4) / 2)


def test_valid_classes_respects_min_positives():
    y, _ = synth()
    y[:, 0] = 0.0
    y[:5, 1], y[5:, 1] = 1.0, 0.0  # only 5 positives
    v = valid_classes(y, min_pos=10)
    assert not v[0] and not v[1]


def test_iou_counts_and_mean_iou():
    pred = np.array([[0, 1, 1, 2]])
    target = np.array([[0, 1, 2, 2]])
    counts = iou_counts(pred, target, 3)
    assert counts[0, 0].tolist() == [1, 1, 1]  # intersection
    assert counts[0, 1].tolist() == [1, 2, 1]  # predicted
    assert counts[0, 2].tolist() == [1, 1, 2]  # target
    assert mean_iou(counts, np.array([True, True, True])) == pytest.approx((1 + 0.5 + 0.5) / 3)


def test_perfect_prediction_has_iou_one():
    p = np.random.default_rng(0).integers(0, 5, (6, 16))
    assert mean_iou(iou_counts(p, p, 5), np.ones(5, bool)) == pytest.approx(1.0)


def _mean(run, idx):
    """Each run is a per-image score vector; the metric is its mean over the sampled images."""
    return float(run[idx].mean())


def test_bootstrap_finds_a_real_difference():
    base = np.random.default_rng(0).normal(0.0, 1.0, 300)
    runs = {"a": [base + 0.5] * 5, "b": [base] * 5}
    d = seed_image_bootstrap(_mean, runs, 300, [("a", "b")], 200)[("a", "b")]
    lo, _, p = summarise_bootstrap(d)
    assert lo > 0 and p < 0.05


def test_bootstrap_of_identical_arms_is_zero():
    rng = np.random.default_rng(0)
    runs = [rng.normal(0.0, 1.0, 300) for _ in range(5)]
    d = seed_image_bootstrap(_mean, {"a": runs, "b": runs}, 300, [("a", "b")], 100)[("a", "b")]
    assert np.allclose(d, 0.0)


def test_seed_noise_widens_the_interval():
    """The reason for resampling seeds: per-arm seed noise must show up in the interval."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, 300)

    def width(noise):
        runs = {"a": [base + e for e in rng.normal(0, noise, 5)],
                "b": [base + e for e in rng.normal(0, noise, 5)]}
        lo, hi, _ = summarise_bootstrap(
            seed_image_bootstrap(_mean, runs, 300, [("a", "b")], 300)[("a", "b")])
        return hi - lo

    assert width(0.5) > width(0.0)


def test_p_value_is_floored_at_bootstrap_resolution():
    assert summarise_bootstrap(np.ones(100))[2] == pytest.approx(0.02)


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
