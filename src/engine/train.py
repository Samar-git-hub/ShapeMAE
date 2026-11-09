"""Pretraining loop, one (arm, seed) per invocation.

    python -m src.engine.train --config configs/arm_c_adaptive.yaml --seed 0
    python -m src.engine.train --config configs/arm_a_random.yaml --benchmark

All arms share this file, this optimiser, this schedule and, within a seed, the same
initialisation and data order. The only thing that differs is how the mask is built.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, apply_cli_overrides, load_config
from ..data.dataset import build_loader, normalize_batch
from ..masking.adaptive import CategoryLossEMA
from ..masking.strategies import build_batch_masks
from ..models.mae import build_mae, count_params
from ..utils import device as dev
from ..utils.logging import RunLogger, fmt_eta
from ..utils.scheduler import apply_lr, lr_at, param_groups_no_decay
from ..utils.seed import seed_everything, step_rng


def resolve_mask_ratio(cfg: Config) -> float:
    """The masking ratio is resolved by curation, not hardcoded. Reading it from the cache
    guarantees every arm uses the same M without anyone having to remember a flag."""
    if cfg.mask.ratio_override is not None:
        return float(cfg.mask.ratio_override)
    meta = Path(cfg.data.cache_dir) / "meta.json"
    if meta.exists():
        baked = json.loads(meta.read_text("utf-8")).get("mask_ratio")
        if baked:
            return float(baked)
    return float(cfg.mask.ratio)


@torch.no_grad()
def evaluate_bank(model, loader, bank: dict, device) -> dict:
    """Validation loss under the frozen mask bank.

    Raw training losses are not comparable across arms: they optimise different mask
    distributions. Every arm is instead scored on identical, pre-generated masks.
    """
    model.eval()
    totals = {"random": 0.0, "instance": 0.0}
    pos = 0
    for batch in loader:
        n = batch["image"].shape[0]
        imgs = normalize_batch(batch["image"], device)
        for key in totals:
            mask = torch.from_numpy(bank[key][pos:pos + n]).to(device)
            with dev.autocast(device):
                loss, _, _ = model(imgs, mask)
            totals[key] += float(loss) * n
        pos += n
    model.train()
    return {f"val_loss_{k}": v / max(pos, 1) for k, v in totals.items()}


def save_atomic(state: dict, path: Path) -> None:
    """A reboot mid-write must never destroy the only resume point."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(cfg: Config, args) -> None:
    seed_everything(cfg.seed)
    device = dev.select_device(cfg.device)
    dev.keep_awake()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt, resume_ckpt = run_dir / "ckpt.pt", run_dir / "resume.pt"
    if final_ckpt.exists() and not args.benchmark:
        print(f"[skip] {run_dir} is already finished")
        return

    resuming = args.resume and resume_ckpt.exists()
    log = RunLogger(run_dir, resume=resuming)
    log.info(f"[run] {cfg.arm} seed {cfg.seed} on {device} "
             f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    cfg.mask.ratio = resolve_mask_ratio(cfg)
    n_masked = cfg.n_masked
    log.info(f"[mask] {cfg.mask.strategy} r={cfg.mask.ratio:.2f} M={n_masked}/{cfg.num_patches}")

    ds_train, dl_train = build_loader(cfg, "train", device)
    ds_val, dl_val = build_loader(cfg, "val", device, shuffle=False, workers=0)
    n_cats = ds_train.n_cats
    log.info(f"[data] train={len(ds_train):,} val={len(ds_val):,} cats={n_cats}")

    model = build_mae(cfg).to(device)
    enc_params, total_params = count_params(model)
    log.info(f"[model] encoder={enc_params/1e6:.2f}M total={total_params/1e6:.2f}M")

    lr = cfg.train.base_lr * cfg.train.batch_size / 256
    optimizer = torch.optim.AdamW(param_groups_no_decay(model, cfg.train.weight_decay), lr=lr,
                                  betas=tuple(cfg.train.betas),
                                  fused=device.type == "cuda")

    ema = CategoryLossEMA(
        n_cats=n_cats, momentum=cfg.adaptive.momentum,
        warmup_epochs=cfg.adaptive.warmup_epochs, total_epochs=cfg.train.epochs,
        tau_start=cfg.adaptive.tau_start, tau_end=cfg.adaptive.tau_end,
        w_min=cfg.adaptive.w_min, w_max=cfg.adaptive.w_max,
        min_patches=cfg.adaptive.min_patches_per_cat, clip_q=cfg.adaptive.loss_clip_q)

    start_epoch = 0
    if resuming:
        state = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        log.drop_from(start_epoch)
        seed_everything(cfg.seed + start_epoch)  # do not replay the first epochs' data order
        log.info(f"[resume] from epoch {start_epoch}")

    cfg.save(run_dir / "config.yaml")

    if args.benchmark:
        run_benchmark(model, dl_train, optimizer, cfg, device, n_masked, n_cats, log)
        return

    from .evaluate import load_or_build_bank
    bank = load_or_build_bank(cfg, ds_val, n_masked)

    steps = max(len(dl_train), 1)
    log.info(f"[train] {cfg.train.epochs} epochs x {steps} steps, batch "
             f"{cfg.train.batch_size}, lr {lr:.2e}\n")

    t_start = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        weights = (ema.weights(epoch) if cfg.adaptive.enabled else np.ones(n_cats, np.float32))
        model.train()
        t_epoch = time.time()
        run = {"loss": 0.0, "n": 0, "inst_frac": 0.0, "n_inst": 0.0, "topup": 0.0, "skipped": 0.0}
        cat_hits = np.zeros(n_cats, np.int64)

        def settle(done: tuple) -> None:
            """Read back a finished step. Per-category attribution runs in EVERY arm: the others
            record it for the figures, only the adaptive arm feeds it back into masking."""
            loss, per_patch, masks_cpu, cat_grid, stats, bs = done
            ema.accumulate(per_patch.float().cpu().numpy(), masks_cpu.numpy(), cat_grid.numpy())
            run["loss"] += float(loss) * bs
            run["n"] += bs
            for k in ("inst_frac", "n_inst", "topup", "skipped"):
                run[k] += stats[k] * bs
            cat_hits[:] += stats["cat_hits"]

        # Reading a value back from the GPU blocks until it is idle, so each step is settled
        # only after the NEXT step's masks are built. That CPU work then overlaps with the
        # GPU instead of leaving it waiting: 20% more throughput for the object arms, and the
        # masks are unchanged because they come from a per-step RNG.
        pending = None
        for step, batch in enumerate(dl_train):
            apply_lr(optimizer, lr_at(epoch + step / steps, lr, cfg.train.warmup_epochs,
                                      cfg.train.epochs, cfg.train.min_lr))

            masks_cpu, stats = build_batch_masks(batch, cfg.mask.strategy, n_masked, weights,
                                                 step_rng(cfg.seed, epoch, step))
            if pending is not None:
                settle(pending)

            imgs = normalize_batch(batch["image"], device)
            masks = masks_cpu.to(device, non_blocking=True)
            with dev.autocast(device, cfg.train.amp):
                loss, _, per_patch = model(imgs, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            pending = (loss.detach(), per_patch.detach(), masks_cpu, batch["cat_grid"], stats,
                       batch["image"].shape[0])

            if step % cfg.train.log_every == 0:
                done = (epoch - start_epoch) * steps + step + 1
                total = (cfg.train.epochs - start_epoch) * steps
                eta = (time.time() - t_start) / done * (total - done)
                log.progress(f"  ep {epoch:3d} [{step:4d}/{steps}] "
                             f"loss {run['loss']/max(run['n'],1):.4f} eta {fmt_eta(eta)}")
        settle(pending)

        cat_stats = ema.end_epoch()
        ema.snapshot(epoch)

        # NaN on skipped epochs rather than carrying a value forward: the figures should plot
        # real measurements only.
        due = epoch % max(cfg.train.eval_every, 1) == 0 or epoch == cfg.train.epochs - 1
        val = (evaluate_bank(model, dl_val, bank, device) if due else
               {"val_loss_random": float("nan"), "val_loss_instance": float("nan")})
        n = max(run["n"], 1)
        elapsed = time.time() - t_epoch
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": run["loss"] / n, **val,
               "inst_frac": run["inst_frac"] / n, "n_inst": run["n_inst"] / n,
               "topup": run["topup"] / n, "skipped": run["skipped"] / n, **cat_stats,
               "epoch_s": round(elapsed, 1), "img_per_s": round(n / max(elapsed, 1e-6), 1),
               "peak_vram_gb": round(dev.peak_vram_gb(device), 2)}
        log.log_epoch(row)
        print(f"\r  ep {epoch:3d}  train {row['train_loss']:.4f}  "
              f"val_rand {row['val_loss_random']:.4f}  val_inst {row['val_loss_instance']:.4f}  "
              f"{row['img_per_s']:.0f} img/s{' ' * 20}", flush=True)

        np.save(run_dir / "cat_ema.npy", ema.history_array())

        if (epoch + 1) % cfg.train.ckpt_every == 0:
            save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "ema": ema.state_dict(), "epoch": epoch}, resume_ckpt)

        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            print(f"\n[stop] after epoch {epoch}; rerun with --resume to continue")
            return

    # The finished run keeps weights only: 26 MB instead of 77 MB, on a nearly full disk.
    save_atomic({"model": model.state_dict(), "epoch": cfg.train.epochs - 1,
                 "mask_ratio": cfg.mask.ratio}, final_ckpt)
    resume_ckpt.unlink(missing_ok=True)
    print(f"\n[done] {cfg.arm} seed {cfg.seed} in {fmt_eta(time.time() - t_start)} -> {run_dir}")


def run_benchmark(model, loader, optimizer, cfg: Config, device, n_masked: int, n_cats: int,
                  log: RunLogger, steps: int = 80) -> None:
    """Measure real throughput before committing to a long run."""
    model.train()
    weights = np.ones(n_cats, np.float32)
    it = iter(loader)
    t0 = seen = 0.0
    pending = None
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        masks_cpu, _ = build_batch_masks(batch, cfg.mask.strategy, n_masked, weights,
                                         step_rng(cfg.seed, 0, step))
        if pending is not None:
            _ = pending.float().cpu()  # settle the previous step, as training does
        imgs = normalize_batch(batch["image"], device)
        masks = masks_cpu.to(device)
        with dev.autocast(device, cfg.train.amp):
            loss, _, per_patch = model(imgs, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pending = per_patch.detach()

        if step == 19:  # discard warmup
            t0, seen = time.time(), 0.0
        elif step > 19:
            seen += batch["image"].shape[0]

    rate = seen / max(time.time() - t0, 1e-6)
    epoch_s = len(loader.dataset) / rate
    log.info(f"\n[bench] {rate:.0f} img/s | peak VRAM {dev.peak_vram_gb(device):.2f} GB")
    log.info(f"[bench] {epoch_s:.1f} s/epoch -> {cfg.train.epochs} epochs = "
             f"{fmt_eta(epoch_s * cfg.train.epochs)} per run")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--stop-after", type=int, help="stop after N epochs, keeping the resume point")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    cfg.seed = args.seed
    train(cfg, args)


if __name__ == "__main__":
    main()
