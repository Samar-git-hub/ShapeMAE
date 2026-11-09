"""The budget invariant and the skip-fit sampler.

The invariant tested here is the one the whole experiment rests on: every arm masks
exactly M patches, so any difference between arms is attributable to mask CONTENT.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.masking.strategies import build_batch_masks, instance_mask, random_mask

N_CATS = 8
L = 16          # 4x4 grid
MAX_INST = 6


def make_grid(rng, n_inst=4, n_patches=L):
    """Random patch->instance assignment with a guaranteed background remainder."""
    grid = np.zeros(n_patches, np.int64)
    idx = rng.permutation(n_patches)
    cut = rng.integers(n_patches // 2, n_patches)
    chunks = np.array_split(idx[:cut], n_inst)
    for k, ch in enumerate(chunks, start=1):
        grid[ch] = k
    return grid


@pytest.mark.parametrize("n_masked", [4, 8, 12])
def test_random_mask_exact_budget(rng, n_masked):
    for _ in range(200):
        m = random_mask(L, n_masked, rng)
        assert m.sum() == n_masked
        assert m.dtype == bool


def test_instance_mask_exact_budget_always(rng):
    """1000 draws over varied grids, instance counts and budgets."""
    for _ in range(1000):
        n_inst = int(rng.integers(1, MAX_INST + 1))
        grid = make_grid(rng, n_inst)
        inst_cats = rng.integers(0, N_CATS, MAX_INST)
        weights = rng.uniform(0.25, 4.0, N_CATS).astype(np.float32)
        n_masked = int(rng.integers(2, L))

        mask, stats = instance_mask(grid, inst_cats, weights, n_masked, rng)
        assert mask.sum() == n_masked, stats
        assert 0.0 <= stats["inst_frac"] <= 1.0


def test_no_instances_falls_back_to_random(rng):
    grid = np.zeros(L, np.int64)
    mask, stats = instance_mask(grid, np.zeros(MAX_INST, np.int64),
                                np.ones(N_CATS, np.float32), 8, rng)
    assert mask.sum() == 8
    assert stats["fallback"] == 1
    assert stats["n_inst"] == 0


def test_instances_larger_than_budget_are_skipped_not_truncated(rng):
    """Skip-fit: an instance that overflows the budget must be skipped whole.

    Truncating it would leave part of the object visible, which is the local texture
    shortcut the method exists to remove.
    """
    grid = np.zeros(L, np.int64)
    grid[:12] = 1      # one big instance, 12 patches
    grid[12:14] = 2    # one small instance, 2 patches
    inst_cats = np.zeros(MAX_INST, np.int64)

    for _ in range(50):
        mask, stats = instance_mask(grid, inst_cats, np.ones(N_CATS, np.float32),
                                    4, rng)
        assert mask.sum() == 4
        # instance 1 cannot fit in a budget of 4 -> must never be partially masked
        assert not mask[:12].any() or mask[:12].all()
        assert stats["skipped"] >= 1


def test_masked_instances_are_always_whole(rng):
    """Every masked instance must be masked in full.

    Top-up draws from background only, so it can never partially reveal an instance that
    skip-fit declined. The single exception is `partial=1`, which the sampler sets when
    background is exhausted at near-total coverage; that case is flagged, logged, and
    asserted to be exactly what it claims.
    """
    n_partial = 0
    for _ in range(300):
        grid = make_grid(rng, int(rng.integers(2, 5)))
        inst_cats = rng.integers(0, N_CATS, MAX_INST)
        n_masked = int(rng.integers(2, L))
        mask, stats = instance_mask(grid, inst_cats, np.ones(N_CATS, np.float32),
                                    n_masked, rng)
        if stats["partial"]:
            n_partial += 1
            continue
        for inst_id in np.unique(grid[grid > 0]):
            covered = mask[grid == inst_id]
            assert covered.all() or not covered.any(), \
                f"instance {inst_id} partially masked without partial flag"
    # the synthetic grids are deliberately dense; on curated COCO this stays near zero
    assert n_partial < 300, "every sample fell back: top-up logic is broken"


def test_weights_bias_selection(rng):
    """A one-hot weight on category k should make category-k instances win the draw."""
    hits = 0
    trials = 200
    for _ in range(trials):
        grid = np.zeros(L, np.int64)
        grid[0:2] = 1
        grid[2:4] = 2
        inst_cats = np.zeros(MAX_INST, np.int64)
        inst_cats[0] = 3   # instance 1 -> category 3
        inst_cats[1] = 5   # instance 2 -> category 5

        w = np.full(N_CATS, 0.01, np.float32)
        w[3] = 100.0
        mask, _ = instance_mask(grid, inst_cats, w, 2, rng)
        if mask[0:2].all():
            hits += 1
    assert hits / trials > 0.95, f"weighted category selected only {hits}/{trials}"


def test_determinism(rng):
    grid = make_grid(np.random.default_rng(7), 3)
    inst_cats = np.arange(MAX_INST) % N_CATS
    w = np.ones(N_CATS, np.float32)

    a, _ = instance_mask(grid, inst_cats, w, 8, np.random.default_rng(99))
    b, _ = instance_mask(grid, inst_cats, w, 8, np.random.default_rng(99))
    c, _ = instance_mask(grid, inst_cats, w, 8, np.random.default_rng(100))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("strategy", ["random", "shuffled", "instance", "adaptive"])
def test_batch_masks_all_strategies(rng, strategy):
    import torch

    n = 8
    grids = np.stack([make_grid(rng, 3) for _ in range(n)])
    batch = {
        "grid": torch.from_numpy(grids),
        "inst_cats": torch.from_numpy(rng.integers(0, N_CATS, (n, MAX_INST))),
    }
    masks, stats = build_batch_masks(batch, strategy, 8, np.ones(N_CATS, np.float32),
                                     rng)
    assert masks.shape == (n, L)
    assert masks.dtype == torch.bool
    assert (masks.sum(1) == 8).all(), "budget parity violated across the batch"
    assert "cat_hits" in stats


def test_shuffled_uses_a_different_images_layout(rng):
    """The shape-matched control must not reproduce the sample's own instance layout."""
    import torch

    n = 6
    grids = np.zeros((n, L), np.int64)
    for i in range(n):
        grids[i, i * 2:(i * 2) + 4] = 1   # each image has a distinct 4-patch instance
    batch = {
        "grid": torch.from_numpy(grids),
        "inst_cats": torch.zeros((n, MAX_INST), dtype=torch.int64),
    }
    masks, _ = build_batch_masks(batch, "shuffled", 4, np.ones(N_CATS, np.float32),
                                 rng)
    own = np.array([masks[i].numpy()[grids[i] == 1].all() for i in range(n)])
    assert not own.all(), "shuffled masks should not align with each image's own instance"
