"""Patch-grid pooling, geometry alignment, and the 1-based instance-id convention.

A silent off-by-one between instance ids and `inst_cats` would mislabel every category
without ever raising, so the convention gets its own tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.grid import cats_for_ids, coverage, instance_sizes, pool_instance_grid


def test_pool_assigns_full_patch():
    inst = np.zeros((32, 32), np.uint8)
    inst[0:16, 0:16] = 3
    grid = pool_instance_grid(inst, patch=16, max_id=8, thresh=0.25)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 3
    assert grid[0, 1] == grid[1, 0] == grid[1, 1] == 0


def test_threshold_controls_halo():
    """A patch that is 25% covered belongs to the instance at thresh<=0.25 and to
    background above it. This is the knob that suppresses silhouette leakage."""
    inst = np.zeros((16, 16), np.uint8)
    inst[0:8, 0:8] = 1  # exactly 25% of the 16x16 patch
    assert pool_instance_grid(inst, 16, 8, thresh=0.25)[0, 0] == 1
    assert pool_instance_grid(inst, 16, 8, thresh=0.50)[0, 0] == 0


def test_background_never_wins_argmax():
    """60% background, 40% instance -> still assigned to the instance at thresh=0.25.
    Plain majority vote would return background and expose the object's outline."""
    inst = np.zeros((16, 16), np.uint8)
    inst[0:6, :] = 2  # 6/16 rows = 37.5%
    assert pool_instance_grid(inst, 16, 8, thresh=0.25)[0, 0] == 2


def test_larger_instance_wins_within_patch():
    inst = np.zeros((16, 16), np.uint8)
    inst[0:10, :] = 1
    inst[10:16, :] = 2
    assert pool_instance_grid(inst, 16, 8, thresh=0.25)[0, 0] == 1


def test_hflip_equivariance():
    """Flipping pixels then pooling == pooling then flipping. If this fails, the
    augmentation silently misaligns masks from images."""
    rng = np.random.default_rng(3)
    inst = rng.integers(0, 5, (64, 64)).astype(np.uint8)
    a = pool_instance_grid(inst[:, ::-1].copy(), 16, 8, 0.25)
    b = pool_instance_grid(inst, 16, 8, 0.25)[:, ::-1]
    assert np.array_equal(a, b)


def test_pool_rejects_bad_shape():
    with pytest.raises(ValueError):
        pool_instance_grid(np.zeros((30, 32), np.uint8), patch=16)


def test_cats_for_ids_is_one_based():
    """inst_cats[k-1] is the category of instance id k. Indexing by loop position
    instead would shift every label by one."""
    inst_cats = np.array([10, 20, 30, 40])
    assert cats_for_ids(np.array([1]), inst_cats)[0] == 10
    assert cats_for_ids(np.array([4]), inst_cats)[0] == 40
    assert np.array_equal(cats_for_ids(np.array([2, 3]), inst_cats), [20, 30])


def test_cats_for_ids_rejects_background():
    with pytest.raises(ValueError):
        cats_for_ids(np.array([0, 1]), np.array([5, 6]))


def test_cats_for_ids_survives_gaps():
    """A crop can leave non-contiguous ids; the lookup must still be correct."""
    inst_cats = np.array([7, 8, 9, 11, 13])
    assert np.array_equal(cats_for_ids(np.array([2, 5]), inst_cats), [8, 13])


def test_cats_for_ids_out_of_range_is_minus_one():
    assert cats_for_ids(np.array([9]), np.array([1, 2]))[0] == -1


def test_coverage_and_sizes():
    grid = np.array([[0, 1], [1, 2]], np.uint8)
    assert coverage(grid) == 0.75
    ids, sizes = instance_sizes(grid)
    assert np.array_equal(ids, [1, 2])
    assert np.array_equal(sizes, [2, 1])
