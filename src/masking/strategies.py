"""Masking policies. All of them honour one invariant:

    EVERY image, EVERY arm, EVERY step masks exactly M = round(ratio * L) patches.

That is more than fairness. Because each row masks the same count, the encoder's visible
sequence length L - M is constant across the batch, which is what makes the gather in
`MaskedAutoencoderViT.forward_encoder` rectangular. Break the invariant and the model raises
rather than silently training on ragged batches.

    random    uniform random patches                        (baseline)
    shuffled  whole instances borrowed from ANOTHER image   (shape-matched control)
    instance  whole instances, uniform category weights
    adaptive  whole instances, EMA-weighted categories

`shuffled` is the control that separates semantics from geometry. Instance masks are large
contiguous blobs and random masks are scattered single patches, so an instance-beats-random
result alone cannot distinguish "objects matter" from "big holes are harder". `shuffled`
reuses a real instance layout from a different image: the blob statistics match exactly while
the alignment with this image's objects is gone.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.grid import cats_for_ids

STRATEGIES = ("random", "shuffled", "instance", "adaptive")


def random_mask(n_patches: int, n_masked: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n_patches, dtype=bool)
    mask[rng.choice(n_patches, size=n_masked, replace=False)] = True
    return mask


def _patch_index_by_instance(grid_flat: np.ndarray):
    """One argsort instead of an equality scan per instance."""
    order = np.argsort(grid_flat, kind="stable")
    sorted_g = grid_flat[order]
    ids = np.unique(sorted_g)
    ids = ids[ids != 0]
    if ids.size == 0:
        return ids, [], np.zeros(0, np.int64)
    starts = np.searchsorted(sorted_g, ids, side="left")
    ends = np.searchsorted(sorted_g, ids, side="right")
    return ids, [order[s:e] for s, e in zip(starts, ends)], (ends - starts).astype(np.int64)


def instance_mask(grid_flat: np.ndarray, inst_cats: np.ndarray, cat_weights: np.ndarray,
                  n_masked: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Mask whole object instances up to an exact budget of `n_masked` patches.

    Selection is Gumbel-top-k over instances, which samples without replacement exactly
    proportional to the weights (Plackett-Luce) in one vectorised step.

    The accumulation is SKIP-FIT, not truncate-on-overshoot: an instance that would overflow
    the remaining budget is skipped and the walk continues. Partially masking an object
    leaves some of its patches visible, handing the model the local-texture shortcut this
    project exists to remove. The leftover budget is topped up from BACKGROUND patches only,
    because any unmasked patch could belong to an instance skip-fit deliberately declined,
    which would partially mask it through the back door.
    """
    n_patches = grid_flat.size
    ids, groups, sizes = _patch_index_by_instance(grid_flat)

    if ids.size == 0:  # no instances survived the crop: fall back, still exactly M
        return random_mask(n_patches, n_masked, rng), {
            "n_inst": 0, "inst_frac": 0.0, "topup": n_masked, "skipped": 0,
            "partial": 0, "masked_cats": [], "fallback": 1}

    cats = cats_for_ids(ids, inst_cats)
    weights = np.where(cats >= 0, cat_weights[np.clip(cats, 0, len(cat_weights) - 1)], 1.0)
    keys = np.log(np.maximum(weights, 1e-12)) + rng.gumbel(size=weights.size)

    mask = np.zeros(n_patches, dtype=bool)
    used = n_inst = skipped = 0
    masked_cats: list[int] = []
    for j in np.argsort(-keys):
        size = int(sizes[j])
        if used + size <= n_masked:
            mask[groups[j]] = True
            used += size
            n_inst += 1
            if cats[j] >= 0:
                masked_cats.append(int(cats[j]))
            if used == n_masked:
                break
        else:
            skipped += 1

    residual = n_masked - used
    partial = 0
    if residual > 0:
        free = np.flatnonzero(~mask & (grid_flat == 0))
        take = min(residual, free.size)
        if take:
            mask[rng.choice(free, size=take, replace=False)] = True
        if take < residual:
            # Background exhausted, only possible at near-total coverage. Fill from anywhere
            # and flag it so the frequency shows in the logs.
            rest = np.flatnonzero(~mask)
            mask[rng.choice(rest, size=residual - take, replace=False)] = True
            partial = 1

    if int(mask.sum()) != n_masked:
        raise AssertionError(f"budget invariant broken: {int(mask.sum())} != {n_masked}")

    return mask, {"n_inst": n_inst, "inst_frac": used / max(n_masked, 1), "topup": residual,
                  "skipped": skipped, "partial": partial, "masked_cats": masked_cats,
                  "fallback": 0}


def build_batch_masks(batch: dict, strategy: str, n_masked: int, cat_weights: np.ndarray,
                      rng: np.random.Generator) -> tuple[torch.Tensor, dict]:
    """Generate masks for a whole batch on the CPU.

    Masks are built in the main process rather than in DataLoader workers because the
    adaptive arm's weights change every epoch and would otherwise have to be shipped into
    worker processes. One code path for all arms is worth the small cost per step.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    grids = batch["grid"].numpy()
    inst_cats = batch["inst_cats"].numpy()
    n, n_patches = grids.shape
    masks = np.zeros((n, n_patches), dtype=bool)
    agg = {"n_inst": 0.0, "inst_frac": 0.0, "topup": 0.0, "skipped": 0.0, "partial": 0.0,
           "fallback": 0.0}
    cat_hits = np.zeros(len(cat_weights), dtype=np.int64)

    if strategy == "random":
        for i in range(n):
            masks[i] = random_mask(n_patches, n_masked, rng)
        stats = {**agg, "topup": float(n_masked), "cat_hits": cat_hits}
        return torch.from_numpy(masks), stats

    weights = cat_weights if strategy == "adaptive" else np.ones_like(cat_weights)
    for i in range(n):
        # `shuffled` borrows a real instance layout from the next image in the batch: identical
        # blob geometry, deliberately unrelated to this image.
        src = (i + 1) % n if (strategy == "shuffled" and n > 1) else i
        masks[i], stats = instance_mask(grids[src], inst_cats[src], weights, n_masked, rng)
        for k in agg:
            agg[k] += float(stats[k])
        for c in stats["masked_cats"]:
            cat_hits[c] += 1

    return torch.from_numpy(masks), {**{k: v / max(n, 1) for k, v in agg.items()},
                                     "cat_hits": cat_hits}
