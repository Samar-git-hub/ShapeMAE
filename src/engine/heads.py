"""Frozen-encoder heads: k-NN, linear, class attention and dense.

    python -m src.engine.heads --arm instance --seed 0              # validation only
    python -m src.engine.heads --arm instance --seed 0 --split test

Every head trains on the 13,500 pretraining images with the encoder frozen, picks its best
epoch on the validation split, and is only then scored. `--split val` never reads the test
set, so everything can be developed against validation; `--split test` is the final read.

Features are extracted with augmentation off, so a score is a deterministic function of the
checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, apply_cli_overrides, load_config
from ..data.dataset import build_loader, normalize_batch
from ..models.heads import ClassAttentionHead, DenseHead
from ..models.mae import build_mae
from ..utils import device as dev
from ..utils.logging import fmt_eta
from ..utils.seed import seed_everything
from .stats import iou_counts, macro_ap, mean_iou, valid_classes


def load_encoder(cfg: Config, arm: str, seed: int, device) -> nn.Module:
    seed_everything(seed)
    model = build_mae(cfg).to(device)
    ckpt = Path(cfg.train.out_dir) / arm / f"seed_{seed}" / "ckpt.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    return model.eval()


@torch.no_grad()
def extract(model, cfg: Config, mode: str, device) -> dict:
    """Tokens (N, L, D) in fp16 on the CPU, plus pooled features, labels and the patch map."""
    ds, loader = build_loader(cfg, mode, device, shuffle=False, augment=False,
                              batch_size=cfg.heads.batch_size, workers=4)
    # One allocation filled in place, pooling on the GPU per batch: tokens are 4 GB for the
    # train split, and on a nearly full disk Windows cannot grow the pagefile to hold a copy.
    tokens = None
    pooled, labels, cat_grid = [], [], []
    pos = 0
    for batch in loader:
        with dev.autocast(device):
            t = model.tokens(normalize_batch(batch["image"], device), cfg.heads.last_k)
        if tokens is None:
            tokens = torch.empty((len(ds), *t.shape[1:]), dtype=torch.float16)
        tokens[pos:pos + t.shape[0]] = t.to(torch.float16).cpu()
        pos += t.shape[0]
        pooled.append(t.float().mean(1).cpu().numpy())
        labels.append(batch["labels"].numpy())
        cat_grid.append(batch["cat_grid"].numpy())
    return {"tokens": tokens, "pooled": np.concatenate(pooled),
            "labels": np.concatenate(labels), "cat_grid": np.concatenate(cat_grid)}


def knn_scores(xtr: np.ndarray, ytr: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """Cosine k-NN with a similarity-weighted label vote. Parameter-free."""
    a = xtr / (np.linalg.norm(xtr, axis=1, keepdims=True) + 1e-8)
    b = xq / (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-8)
    out = np.zeros((len(b), ytr.shape[1]), np.float32)
    for i in range(0, len(b), 512):  # chunked: the full similarity matrix is large
        sim = b[i:i + 512] @ a.T
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.exp(np.take_along_axis(sim, idx, axis=1) / 0.07)
        out[i:i + 512] = np.einsum("nk,nkc->nc", w, ytr[idx]) / w.sum(1, keepdims=True)
    return out


class Standardise:
    """Per-dimension z-score with statistics from the training split only."""

    def __init__(self, x: np.ndarray, device) -> None:
        self.mu = torch.from_numpy(x.mean(0)).float().to(device)
        self.sd = torch.from_numpy(x.std(0) + 1e-6).float().to(device)

    @classmethod
    def from_tokens(cls, tokens: torch.Tensor, device) -> "Standardise":
        """Statistics over every token of every image, streamed through the GPU in chunks.
        Tokens vary far more than their per-image means, so they need their own scale."""
        std = cls.__new__(cls)
        total = torch.zeros(tokens.shape[-1], dtype=torch.float64, device=device)
        squares = torch.zeros_like(total)
        for chunk in tokens.split(1024):
            x = chunk.to(device).flatten(0, 1).double()
            total += x.sum(0)
            squares += (x * x).sum(0)
        n = tokens.shape[0] * tokens.shape[1]
        mean = total / n
        std.mu = mean.float()
        std.sd = ((squares / n - mean * mean).clamp_min(0).sqrt() + 1e-6).float()
        return std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu) / self.sd


def fit(head: nn.Module, groups: list[dict], loss_fn, forward_val, score_val, n_train: int,
        batch_size: int, epochs: int, weight_decay: float, device) -> tuple[nn.Module, float]:
    """Train `head`, returning the state from its best validation epoch.

    `groups` are the optimizer's parameter groups, each with its own learning rate.
    `loss_fn(head, idx)` is the loss on the training rows `idx`. `forward_val(head)` gives
    validation predictions, and `score_val` turns them into a higher-is-better number.
    """
    head.to(device)
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        for idx in torch.randperm(n_train).split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss_fn(head, idx).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            score = score_val(forward_val(head))
        if score > best:
            best, best_state = score, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval(), best


def batched(fn, tokens: torch.Tensor, std: Standardise, device, chunk: int = 256) -> torch.Tensor:
    """Apply a token head to CPU-resident tokens in chunks."""
    return torch.cat([fn(std(t.to(device, non_blocking=True))).float().cpu()
                      for t in tokens.split(chunk)])


def run_heads(cfg: Config, arm: str, seed: int, split: str, device) -> None:
    run_dir = Path(cfg.train.out_dir) / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    h = cfg.heads
    t0 = time.time()

    model = load_encoder(cfg, arm, seed, device)
    tr = extract(model, cfg, "train", device)
    va = extract(model, cfg, "val", device)
    te = extract(model, cfg, "test", device) if split == "test" else None
    del model
    torch.cuda.empty_cache()
    seed_everything(seed)

    eval_sets = {"val": va} if te is None else {"val": va, "test": te}
    n_cats = tr["labels"].shape[1]
    dim = tr["pooled"].shape[1]
    n_train = len(tr["labels"])

    # one fixed class set per evaluation split, identical for every arm and head
    valid = {k: valid_classes(v["labels"]) for k, v in eval_sets.items()}
    results: dict[str, dict] = {k: {} for k in eval_sets}
    scores: dict[str, np.ndarray] = {}

    def record(name: str, preds: dict[str, np.ndarray]) -> None:
        for k, p in preds.items():
            results[k][name] = macro_ap(eval_sets[k]["labels"], p, valid[k])
        if te is not None:
            scores[name] = preds["test"].astype(np.float16)
        print(f"  {name:<12} " + "  ".join(f"{k} {results[k][name]:.4f}" for k in preds),
              flush=True)

    # ---- k-NN: raw cosine, and cosine after the same z-score the linear head gets --------
    std = Standardise(tr["pooled"], device)
    mu, sd = std.mu.cpu().numpy(), std.sd.cpu().numpy()
    record("knn_raw", {k: knn_scores(tr["pooled"], tr["labels"], v["pooled"], cfg.eval.knn_k)
                       for k, v in eval_sets.items()})
    record("knn_std", {k: knn_scores((tr["pooled"] - mu) / sd, tr["labels"],
                                     (v["pooled"] - mu) / sd, cfg.eval.knn_k)
                       for k, v in eval_sets.items()})

    # ---- pooled heads: features live on the GPU ------------------------------------------
    xtr = std(torch.from_numpy(tr["pooled"]).to(device))
    ytr = torch.from_numpy(tr["labels"]).float().to(device)
    xev = {k: std(torch.from_numpy(v["pooled"]).to(device)) for k, v in eval_sets.items()}
    bce = nn.BCEWithLogitsLoss()

    def pooled_head(name, head, epochs, lr):
        def loss(m, idx):
            idx = idx.to(device)
            return bce(m(xtr[idx]), ytr[idx])

        def score(p):
            return macro_ap(va["labels"], p, valid["val"])

        head, _ = fit(head, [{"params": head.parameters(), "lr": lr}], loss,
                      lambda m: m(xev["val"]).float().cpu().numpy(), score, n_train, 1024, epochs,
                      h.weight_decay, device)
        with torch.no_grad():
            record(name, {k: head(x).float().cpu().numpy() for k, x in xev.items()})

    pooled_head("linear", nn.Linear(dim, n_cats), h.linear_epochs, h.linear_lr)

    # ---- token heads: tokens stay on the CPU and stream through in batches ---------------
    tok_std = Standardise.from_tokens(tr["tokens"], device)
    val_tok = va["tokens"].to(device)

    def attn_loss(m, idx):
        x = tok_std(tr["tokens"][idx].to(device, non_blocking=True))
        return bce(m(x), ytr[idx.to(device)])

    # `where` overfits 13,500 images quickly, so it learns far more slowly than `what`.
    attn = ClassAttentionHead(dim, n_cats)
    attn, _ = fit(attn, [{"params": attn.what.parameters(), "lr": h.attn_what_lr},
                         {"params": attn.where.parameters(), "lr": h.attn_where_lr}], attn_loss,
                  lambda m: batched(m, val_tok, tok_std, device).numpy(),
                  lambda p: macro_ap(va["labels"], p, valid["val"]),
                  n_train, h.batch_size, h.attn_epochs, h.weight_decay, device)
    with torch.no_grad():
        record("class_attn", {k: batched(attn, v["tokens"], tok_std, device).numpy()
                              for k, v in eval_sets.items()})
    torch.save({"head": attn.state_dict(), "mu": tok_std.mu, "sd": tok_std.sd},
               run_dir / "class_attn.pt")

    # ---- dense head: per-patch category, mean IoU over classes present -------------------
    gtr = torch.from_numpy(tr["cat_grid"]).long().to(device)
    ce = nn.CrossEntropyLoss()
    dense_valid = {k: (np.bincount(v["cat_grid"].ravel(), minlength=n_cats + 1) >= 50)
                   for k, v in eval_sets.items()}

    def dense_loss(m, idx):
        x = tok_std(tr["tokens"][idx].to(device, non_blocking=True))
        return ce(m(x).flatten(0, 1), gtr[idx.to(device)].flatten())

    def dense_pred(m, tokens):
        return torch.cat([m(tok_std(t.to(device))).argmax(-1).cpu() for t in tokens.split(256)]
                         ).numpy()

    def dense_score(pred):
        return mean_iou(iou_counts(pred, va["cat_grid"], n_cats + 1), dense_valid["val"])

    dense = DenseHead(dim, n_cats)
    dense, _ = fit(dense, [{"params": dense.parameters(), "lr": h.dense_lr}], dense_loss,
                   lambda m: dense_pred(m, val_tok), dense_score, n_train, h.batch_size,
                   h.dense_epochs, h.weight_decay, device)
    with torch.no_grad():
        for k, v in eval_sets.items():
            counts = iou_counts(dense_pred(dense, v["tokens"]), v["cat_grid"], n_cats + 1)
            results[k]["dense_miou"] = mean_iou(counts, dense_valid[k])
            if k == "test":
                scores["dense_counts"] = counts
    print("  dense_miou   " + "  ".join(f"{k} {results[k]['dense_miou']:.4f}" for k in eval_sets),
          flush=True)

    # Merge rather than overwrite: finetune.py records its rows in the same files.
    for k in eval_sets:
        path = run_dir / f"heads_{k}.json"
        merged = json.loads(path.read_text("utf-8")) if path.exists() else {}
        path.write_text(json.dumps({**merged, **results[k]}, indent=1), "utf-8")
    if te is not None:
        np.savez_compressed(run_dir / "scores_test.npz", **scores)
    print(f"[heads] {arm} seed {seed} in {fmt_eta(time.time() - t0)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    device = dev.select_device(cfg.device)
    dev.keep_awake()
    print(f"[heads] {args.arm} seed {args.seed} split={args.split}", flush=True)
    run_heads(cfg, args.arm, args.seed, args.split, device)


if __name__ == "__main__":
    main()
