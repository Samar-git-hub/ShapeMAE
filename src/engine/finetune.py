"""Fine-tuning: does a pretraining arm's advantage survive once the weights may move?

    python -m src.engine.finetune --arm instance --seed 0
    python -m src.engine.finetune --arm instance --seed 0 --split test

The classifier is the plain one: final LayerNorm, mean-pool the patch tokens, one linear layer,
with the whole encoder trainable. Training uses the same crop and flip augmentation as
pretraining. The best epoch is picked on validation, and as in `heads.py`, `--split val` never
reads the test set.
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
from ..utils import device as dev
from ..utils.logging import fmt_eta
from ..utils.scheduler import apply_lr, lr_at, param_groups_no_decay
from ..utils.seed import seed_everything
from .heads import load_encoder
from .stats import macro_ap, valid_classes


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.norm.normalized_shape[0], n_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encoder.norm(self.encoder.tokens(imgs, last_k=1)).mean(dim=1))


def freeze_decoder(model: Classifier) -> None:
    """The decoder only exists for pretraining; everything else trains."""
    for name, p in model.encoder.named_parameters():
        if name.startswith(("decoder", "mask_token")):
            p.requires_grad_(False)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out, labels = [], []
    for batch in loader:
        with dev.autocast(device):
            out.append(model(normalize_batch(batch["image"], device)).float().cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(out), np.concatenate(labels)


def finetune(cfg: Config, arm: str, seed: int, split: str, device) -> None:
    h = cfg.heads
    run_dir = Path(cfg.train.out_dir) / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    encoder = load_encoder(cfg, arm, seed, device)
    seed_everything(seed)
    ds, dl_train = build_loader(cfg, "train", device, batch_size=h.ft_batch_size)
    _, dl_val = build_loader(cfg, "val", device, shuffle=False, batch_size=256)

    model = Classifier(encoder, ds.n_cats).to(device)
    freeze_decoder(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft] {arm} seed {seed}: {n_trainable/1e6:.2f}M trainable", flush=True)

    lr = h.ft_lr * h.ft_batch_size / 256
    opt = torch.optim.AdamW(param_groups_no_decay(model, h.ft_weight_decay), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    steps = len(dl_train)
    best, best_state, best_epoch = -1.0, None, -1
    log = []

    for epoch in range(h.ft_epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(dl_train):
            apply_lr(opt, lr_at(epoch + step / steps, lr, h.ft_warmup_epochs, h.ft_epochs))
            with dev.autocast(device):
                loss = bce(model(normalize_batch(batch["image"], device)).float(),
                           batch["labels"].to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())

        pred, y = predict(model, dl_val, device)
        score = macro_ap(y, pred, valid_classes(y))
        log.append({"epoch": epoch, "train_bce": total / steps, "val_map": score})
        print(f"  ep {epoch:3d} bce {total/steps:.4f} val mAP {score:.4f}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    results = {"val": {"finetune": best}}
    if split == "test":
        _, dl_test = build_loader(cfg, "test", device, shuffle=False, batch_size=256)
        pred, y = predict(model, dl_test, device)
        results["test"] = {"finetune": macro_ap(y, pred, valid_classes(y))}
        np.savez_compressed(run_dir / "scores_finetune.npz", scores=pred.astype(np.float16))

    for k, res in results.items():
        path = run_dir / f"heads_{k}.json"
        merged = json.loads(path.read_text("utf-8")) if path.exists() else {}
        path.write_text(json.dumps({**merged, **res}, indent=1), "utf-8")
    (run_dir / "finetune_log.json").write_text(json.dumps(log), "utf-8")
    print(f"[ft] best val {best:.4f} at epoch {best_epoch}"
          + (f", test {results['test']['finetune']:.4f}" if split == "test" else "")
          + f" in {fmt_eta(time.time() - t0)}", flush=True)


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
    finetune(cfg, args.arm, args.seed, args.split, device)


if __name__ == "__main__":
    main()
