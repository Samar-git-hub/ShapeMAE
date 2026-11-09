"""Adaptive category-difficulty controller for arm C.

Each design choice below fixes a specific failure mode:

  * EPOCH-LEVEL updates, not per-step. With a per-step EMA at fixed momentum, the memory
    horizon is 100 *observations of that category*: about one epoch for `person`, but
    dozens of epochs for `toaster`. Since loss falls monotonically, a stale EMA reads as
    high loss, so rare categories get up-weighted for being rare rather than for being
    hard. Updating once per epoch equalises the update rate across categories.

  * RELATIVE difficulty, not absolute loss. The tracked quantity is
    (category mean loss) / (global mean loss) for that epoch. Global loss decay then
    cancels exactly, so staleness can only mean "stale relative difficulty", a far more
    benign error than "stale absolute loss".

  * WINSORISED aggregation. Normalised-pixel loss divides by per-patch standard deviation,
    so a nearly flat patch amplifies sensor and JPEG noise into a large value. The
    training objective keeps the standard MAE mean, but the controller's difficulty
    estimate clips at a high quantile so a handful of degenerate patches cannot steer the
    masking policy.

  * CLIPPED weights. Bounding the odds ratio at w_max/w_min stops a single pathological
    category from monopolising the mask.
"""
from __future__ import annotations

import numpy as np


class CategoryLossEMA:
    def __init__(self, n_cats: int = 80, momentum: float = 0.90, warmup_epochs: int = 20,
                 total_epochs: int = 400, tau_start: float = 0.0, tau_end: float = 1.0,
                 w_min: float = 0.25, w_max: float = 4.0, min_patches: int = 32,
                 clip_q: float = 0.99) -> None:
        self.n_cats = int(n_cats)
        self.momentum = float(momentum)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.min_patches = int(min_patches)
        self.clip_q = float(clip_q)

        self.ema = np.ones(self.n_cats, np.float64)   # relative difficulty, centred on 1
        self.seen = np.zeros(self.n_cats, bool)
        self.updates = np.zeros(self.n_cats, np.int64)

        self._sum = np.zeros(self.n_cats, np.float64)
        self._cnt = np.zeros(self.n_cats, np.int64)
        self.history: list[np.ndarray] = []

    # ---------------------------------------------------------------------------------
    def accumulate(self, per_patch: np.ndarray, mask: np.ndarray,
                   cat_grid: np.ndarray) -> None:
        """Fold one batch into the epoch accumulator.

        per_patch (N, L) float, mask (N, L) bool, cat_grid (N, L) int with 0 = background
        and category+1 otherwise.
        """
        sel = mask & (cat_grid > 0)
        if not sel.any():
            return
        cats = cat_grid[sel] - 1
        losses = per_patch[sel].astype(np.float64)

        if 0.0 < self.clip_q < 1.0 and losses.size > 16:
            losses = np.minimum(losses, np.quantile(losses, self.clip_q))

        self._sum += np.bincount(cats, weights=losses, minlength=self.n_cats)
        self._cnt += np.bincount(cats, minlength=self.n_cats)

    def end_epoch(self) -> dict:
        """Fold the epoch accumulator into the EMA and reset. Returns epoch diagnostics."""
        observed = self._cnt >= self.min_patches
        stats = {"cats_observed": int(observed.sum()), "cat_cv": float("nan"),
                 "cat_mean": float("nan"), "cat_p90": float("nan")}

        if observed.any():
            means = self._sum[observed] / self._cnt[observed]
            global_mean = self._sum[observed].sum() / max(self._cnt[observed].sum(), 1)
            relative = means / max(global_mean, 1e-12)

            idx = np.flatnonzero(observed)
            fresh = ~self.seen[idx]
            # First sighting initialises directly, so there is no bias-correction lag.
            self.ema[idx[fresh]] = relative[fresh]
            old = idx[~fresh]
            self.ema[old] = (self.momentum * self.ema[old]
                             + (1.0 - self.momentum) * relative[~fresh])
            self.seen[idx] = True
            self.updates[idx] += 1

            stats.update(
                cat_mean=float(means.mean()),
                # Spread of difficulty across categories; adaptivity should compress it.
                cat_cv=float(means.std() / max(means.mean(), 1e-12)),
                cat_p90=float(np.percentile(means, 90)),
            )

        self._sum[:] = 0.0
        self._cnt[:] = 0
        return stats

    # ---------------------------------------------------------------------------------
    def tau(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.tau_start
        span = max(self.total_epochs - self.warmup_epochs, 1)
        p = min(max((epoch - self.warmup_epochs) / span, 0.0), 1.0)
        return self.tau_start + p * (self.tau_end - self.tau_start)

    def weights(self, epoch: int) -> np.ndarray:
        """Sampling weight per category. Uniform during warmup: early in training every
        category has high loss and the ranking is noise, so steering on it is worse than
        not steering."""
        w = np.ones(self.n_cats, np.float32)
        if epoch < self.warmup_epochs or not self.seen.any():
            return w
        ref = self.ema[self.seen].mean()
        t = self.tau(epoch)
        w[self.seen] = np.clip((self.ema[self.seen] / max(ref, 1e-12)) ** t,
                               self.w_min, self.w_max).astype(np.float32)
        return w

    def snapshot(self, epoch: int) -> None:
        """Record (ema, weights) for the category-by-epoch heatmap."""
        self.history.append(np.stack([self.ema.copy(),
                                      self.weights(epoch).astype(np.float64)]))

    def history_array(self) -> np.ndarray:
        """(epochs, 2, n_cats): [:, 0] is the EMA, [:, 1] the weights."""
        if not self.history:
            return np.zeros((0, 2, self.n_cats))
        return np.stack(self.history)

    # ---------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"ema": self.ema, "seen": self.seen, "updates": self.updates,
                "history": self.history_array()}

    def load_state_dict(self, state: dict) -> None:
        self.ema = np.asarray(state["ema"], np.float64)
        self.seen = np.asarray(state["seen"], bool)
        self.updates = np.asarray(state["updates"], np.int64)
        hist = np.asarray(state.get("history", np.zeros((0, 2, self.n_cats))))
        self.history = [h for h in hist] if hist.size else []
