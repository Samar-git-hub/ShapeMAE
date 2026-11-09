"""The adaptive controller's arithmetic.

The scale-invariance test is the important one: it is what stops the controller from
tracking the global loss decay instead of relative category difficulty.
"""
from __future__ import annotations

import numpy as np

from src.masking.adaptive import CategoryLossEMA

N_CATS = 6


def feed(ema: CategoryLossEMA, per_cat: dict[int, float], n_patches: int = 64) -> None:
    """One synthetic epoch where each listed category gets `n_patches` at a fixed loss."""
    n_rows = len(per_cat)
    per_patch = np.zeros((n_rows, n_patches))
    mask = np.ones((n_rows, n_patches), bool)
    cat_grid = np.zeros((n_rows, n_patches), np.int64)
    for row, (cat, loss) in enumerate(per_cat.items()):
        per_patch[row] = loss
        cat_grid[row] = cat + 1
    ema.accumulate(per_patch, mask, cat_grid)


def test_constant_loss_converges():
    ema = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    for _ in range(40):
        feed(ema, {0: 2.0, 1: 1.0})
        ema.end_epoch()
    # relative difficulty: cat0 is 2/1.5 of the global mean, cat1 is 1/1.5
    assert np.isclose(ema.ema[0], 2.0 / 1.5, atol=1e-3)
    assert np.isclose(ema.ema[1], 1.0 / 1.5, atol=1e-3)


def test_scale_invariance():
    """Scaling every category's loss by 10x must leave the weights unchanged.

    This is the fix for the frequency/staleness artefact: the controller tracks relative
    difficulty, so global loss decay cancels exactly.
    """
    a = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    b = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    for _ in range(20):
        feed(a, {0: 2.0, 1: 1.0, 2: 0.5})
        a.end_epoch()
        feed(b, {0: 20.0, 1: 10.0, 2: 5.0})
        b.end_epoch()
    assert np.allclose(a.weights(10), b.weights(10), atol=1e-6)


def test_uniform_during_warmup():
    ema = CategoryLossEMA(N_CATS, momentum=0.9, warmup_epochs=5, min_patches=1)
    for _ in range(3):
        feed(ema, {0: 9.0, 1: 0.1})
        ema.end_epoch()
    assert np.allclose(ema.weights(epoch=2), 1.0)
    assert not np.allclose(ema.weights(epoch=100), 1.0)


def test_weights_respect_clip():
    ema = CategoryLossEMA(N_CATS, momentum=0.1, warmup_epochs=0, total_epochs=10,
                          w_min=0.5, w_max=2.0, min_patches=1)
    for _ in range(30):
        feed(ema, {0: 1000.0, 1: 0.001})
        ema.end_epoch()
    w = ema.weights(epoch=10)
    assert w.min() >= 0.5 - 1e-6
    assert w.max() <= 2.0 + 1e-6


def test_harder_category_gets_more_weight():
    ema = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, total_epochs=10,
                          min_patches=1)
    for _ in range(20):
        feed(ema, {0: 3.0, 1: 1.0})
        ema.end_epoch()
    w = ema.weights(epoch=10)
    assert w[0] > w[1]


def test_winsorisation_limits_outliers():
    """A few enormous patch losses must not dominate a category's difficulty estimate.
    norm_pix_loss divides by per-patch std, so flat patches can produce huge values."""
    clipped = CategoryLossEMA(N_CATS, momentum=0.0, warmup_epochs=0, min_patches=1,
                              clip_q=0.90)
    raw = CategoryLossEMA(N_CATS, momentum=0.0, warmup_epochs=0, min_patches=1, clip_q=0.0)

    per_patch = np.ones((1, 100))
    per_patch[0, :5] = 1000.0
    mask = np.ones((1, 100), bool)
    cat_grid = np.ones((1, 100), np.int64)  # all category 0

    for ema in (clipped, raw):
        ema.accumulate(per_patch, mask, cat_grid)
        ema.end_epoch()
    # single observed category -> relative == 1 for both; compare the raw accumulation
    assert clipped.ema[0] == raw.ema[0] == 1.0

    # with two categories the outlier's influence becomes visible
    clipped2 = CategoryLossEMA(N_CATS, momentum=0.0, warmup_epochs=0, min_patches=1,
                               clip_q=0.90)
    raw2 = CategoryLossEMA(N_CATS, momentum=0.0, warmup_epochs=0, min_patches=1, clip_q=0.0)
    pp = np.ones((2, 100))
    pp[0, :5] = 1000.0
    cg = np.stack([np.ones(100, np.int64), np.full(100, 2, np.int64)])
    for ema in (clipped2, raw2):
        ema.accumulate(pp, np.ones((2, 100), bool), cg)
        ema.end_epoch()
    assert clipped2.ema[0] < raw2.ema[0]


def test_state_roundtrip():
    ema = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    for e in range(5):
        feed(ema, {0: 2.0, 1: 1.0})
        ema.end_epoch()
        ema.snapshot(e)

    restored = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    restored.load_state_dict(ema.state_dict())
    assert np.allclose(restored.ema, ema.ema)
    assert restored.history_array().shape == ema.history_array().shape


def test_history_shape():
    ema = CategoryLossEMA(N_CATS, momentum=0.5, warmup_epochs=0, min_patches=1)
    for e in range(4):
        feed(ema, {0: 1.0})
        ema.end_epoch()
        ema.snapshot(e)
    assert ema.history_array().shape == (4, 2, N_CATS)
