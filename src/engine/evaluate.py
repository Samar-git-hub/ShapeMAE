"""Reconstruction, attention distance and the treatment check, on frozen masks.

    python -m src.engine.evaluate --arm instance --seed 0
    python -m src.engine.evaluate --treatment

The key property: every arm is scored on byte-identical masks, generated once with a fixed
seed and UNIFORM category weights. Scoring the adaptive arm on masks from its own learned
policy would be testing on its training distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config, apply_cli_overrides, load_config
from ..data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CachedCocoInstances,
    build_loader,
    normalize_batch,
)
from ..masking.strategies import build_batch_masks, instance_mask, random_mask
from ..utils import device as dev
from .heads import load_encoder
from .train import resolve_mask_ratio


def load_or_build_bank(cfg: Config, ds_val: CachedCocoInstances, n_masked: int) -> dict:
    path = Path(cfg.eval.mask_bank)
    if path.exists():
        data = np.load(path)
        if (data["random"].shape == (len(ds_val), cfg.num_patches)
                and int(data["n_masked"]) == n_masked):
            return {"random": data["random"], "instance": data["instance"]}
        print(f"[bank] stale ({path}); rebuilding")

    rng = np.random.default_rng(cfg.eval.bank_seed)
    n, ell = len(ds_val), cfg.num_patches
    banks = {"random": np.zeros((n, ell), bool), "instance": np.zeros((n, ell), bool)}
    uniform = np.ones(ds_val.n_cats, np.float32)
    for i in range(n):
        sample = ds_val[i]
        banks["random"][i] = random_mask(ell, n_masked, rng)
        banks["instance"][i], _ = instance_mask(sample["grid"].numpy(),
                                                sample["inst_cats"].numpy(), uniform,
                                                n_masked, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_masked=n_masked, **banks)
    print(f"[bank] built {n} x {ell}, M={n_masked}")
    return banks


def denorm_pred(model, imgs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Map a normalised-pixel prediction back to patchified ImageNet-normalised space."""
    target = model.patchify(imgs)
    mu = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    return pred * (var + 1e-6).sqrt() + mu


def patches_to_01(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Undo ImageNet normalisation on a patchified tensor. The channel is the innermost axis
    of a patch, so reshape to (..., 3), rescale, clamp and reshape back."""
    b, ell, _ = patches.shape
    x = patches.reshape(b, ell, patch_size * patch_size, 3)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)
    std = torch.tensor(IMAGENET_STD, device=x.device)
    return (x * std + mean).clamp(0.0, 1.0).reshape(b, ell, -1)


@torch.no_grad()
def score_arm(model, dl_val, bank: dict, device, n_cats: int) -> tuple[dict, np.ndarray]:
    """Masked-patch MSE and PSNR under both banks, plus per-category MSE on instance masks."""
    out: dict[str, float] = {}
    cat_sum, cat_cnt = np.zeros(n_cats), np.zeros(n_cats)

    for key in ("random", "instance"):
        tot_norm = tot_raw = tot_n = 0.0
        pos = 0
        for batch in dl_val:
            n = batch["image"].shape[0]
            imgs = normalize_batch(batch["image"], device)
            mask = torch.from_numpy(bank[key][pos:pos + n]).to(device)
            with dev.autocast(device):
                _, pred, per_patch = model(imgs, mask)

            # raw-pixel error in true [0, 1] units, which is what makes PSNR meaningful
            pred_raw = patches_to_01(denorm_pred(model, imgs, pred.float()), model.patch_size)
            tgt_raw = patches_to_01(model.patchify(imgs), model.patch_size)
            m = mask.float()
            tot_raw += float(((pred_raw - tgt_raw).pow(2).mean(-1) * m).sum())
            tot_norm += float((per_patch.float() * m).sum())
            tot_n += float(m.sum())

            if key == "instance":
                sel = mask.cpu().numpy() & (batch["cat_grid"].numpy() > 0)
                if sel.any():
                    cats = batch["cat_grid"].numpy()[sel] - 1
                    pp = per_patch.float().cpu().numpy()[sel]
                    cat_sum += np.bincount(cats, weights=pp, minlength=n_cats)
                    cat_cnt += np.bincount(cats, minlength=n_cats)
            pos += n

        out[f"mse_{key}"] = tot_norm / max(tot_n, 1)
        out[f"psnr_{key}"] = float(10 * np.log10(1.0 / max(tot_raw / max(tot_n, 1), 1e-12)))

    valid = cat_cnt >= 50
    cat_mse = np.where(valid, cat_sum / np.maximum(cat_cnt, 1), np.nan)
    finite = cat_mse[np.isfinite(cat_mse)]
    out["cat_cv"] = float(finite.std() / max(finite.mean(), 1e-12)) if finite.size else float("nan")
    return out, cat_mse


@torch.no_grad()
def attention_distances(model, ds_val, bank: dict, device, n_images: int = 256) -> dict:
    """Mean attention distance per layer, on all tokens and at the pretraining length."""
    n = min(n_images, len(ds_val))
    imgs = normalize_batch(torch.stack([ds_val[i]["image"] for i in range(n)]), device)
    masks = torch.from_numpy(bank["random"][:n]).to(device)
    full, masked = [], []
    for i in range(0, n, 32):
        full.append(model.attention_distance(imgs[i:i + 32], None))
        masked.append(model.attention_distance(imgs[i:i + 32], masks[i:i + 32]))
    return {"attn_dist_full": torch.stack(full).mean(0).tolist(),
            "attn_dist_masked": torch.stack(masked).mean(0).tolist()}


def evaluate_run(cfg: Config, arm: str, seed: int, device) -> None:
    run_dir = Path(cfg.train.out_dir) / arm / f"seed_{seed}"
    ds_val, dl_val = build_loader(cfg, "val", device, shuffle=False, batch_size=128)
    bank = load_or_build_bank(cfg, ds_val, cfg.n_masked)
    model = load_encoder(cfg, arm, seed, device)

    metrics, cat_mse = score_arm(model, dl_val, bank, device, ds_val.n_cats)
    metrics.update(attention_distances(model, ds_val, bank, device))
    (run_dir / "eval.json").write_text(json.dumps(metrics, indent=1), "utf-8")
    np.save(run_dir / "cat_mse.npy", cat_mse)
    print(f"[eval] {arm:10s} seed {seed} mse_rand={metrics['mse_random']:.4f} "
          f"mse_inst={metrics['mse_instance']:.4f} cv={metrics['cat_cv']:.3f}", flush=True)


def treatment_table(cfg: Config, out_path: Path, n_images: int = 1024) -> pd.DataFrame:
    """What each arm actually hides, measured rather than assumed.

    For each masking strategy, on validation images and independent of any model:
    how often an object is hidden whole, hidden in part or left visible, and what share of
    the masked patches land on objects at all. This tests the premises the design rests on:
    that random masking leaves part of nearly every object visible, and that the donor masks
    of the shape-matched control do not line up with this image's objects.
    """
    ds = CachedCocoInstances(cfg, "val")
    n = min(n_images, len(ds))
    rows = [ds[i] for i in range(n)]
    batch = {k: torch.stack([r[k] for r in rows]) for k in ("grid", "inst_cats")}
    grids = batch["grid"].numpy()
    weights = np.ones(ds.n_cats, np.float32)
    rng = np.random.default_rng(cfg.eval.bank_seed)

    table = []
    for strategy in ("random", "shuffled", "instance"):
        masks = np.zeros(grids.shape, bool)
        for start in range(0, n, 128):  # donors come from within a batch, as in training
            part = {k: v[start:start + 128] for k, v in batch.items()}
            m, _ = build_batch_masks(part, strategy, cfg.n_masked, weights, rng)
            masks[start:start + 128] = m.numpy()

        whole = some = none = 0
        on_object = 0
        for mask, grid in zip(masks, grids):
            on_object += int((mask & (grid > 0)).sum())
            for k in np.unique(grid[grid > 0]):
                hidden = mask[grid == k].mean()
                whole += hidden == 1.0
                none += hidden == 0.0
                some += 0.0 < hidden < 1.0
        total = max(whole + some + none, 1)
        table.append({"strategy": strategy, "object_fully_hidden": whole / total,
                      "object_partly_hidden": some / total, "object_fully_visible": none / total,
                      "masked_patches_on_objects": on_object / masks.sum()})
    df = pd.DataFrame(table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--treatment", action="store_true")
    ap.add_argument("--results", default="results")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    cfg.mask.ratio = resolve_mask_ratio(cfg)
    if args.treatment:
        treatment_table(cfg, Path(args.results) / "treatment.csv")
        return
    evaluate_run(cfg, args.arm, args.seed, dev.select_device(cfg.device))


if __name__ == "__main__":
    main()
