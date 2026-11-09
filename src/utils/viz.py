"""Figures for the README. Everything is drawn from the files in runs/ and results/.

    python -m src.utils.viz --align-check     # run this BEFORE any training, then look at it
    python -m src.utils.viz --all

Each arm keeps one colour across every figure. Multi-seed results are shown as the mean with a
band or the individual seeds, so the spread between runs is always visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from ..config import Config, apply_cli_overrides, load_config  # noqa: E402
from ..data.dataset import CachedCocoInstances, denormalize, normalize_batch  # noqa: E402
from ..engine.analyze import ARMS, FAMILY, HEADS, seeds_of  # noqa: E402
from ..masking.strategies import build_batch_masks  # noqa: E402

ARM_COLOR = {"random": "#2a78d6", "shuffled": "#eb6834", "instance": "#1baf7a",
             "adaptive": "#eda100"}
ARM_LABEL = {"random": "A: random", "shuffled": "A2: donor blobs", "instance": "B: own objects",
             "adaptive": "C: adaptive"}
HEAD_LABEL = {"knn_raw": "k-NN, raw features", "knn_std": "k-NN, standardised",
              "linear": "linear", "class_attn": "class attention",
              "dense_miou": "dense (mIoU)", "finetune": "fine-tuned"}
CONTRAST_LABEL = {("shuffled", "random"): "Geometry\nA2 minus A",
                  ("instance", "shuffled"): "Semantics\nB minus A2",
                  ("adaptive", "instance"): "Adaptivity\nC minus B"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("weights", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 1.8,
})


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _logs(runs: Path, arm: str) -> list[pd.DataFrame]:
    return [pd.read_csv(p) for p in sorted((runs / arm).glob("seed_*/log.csv"))]


def _band(ax, frames: list[pd.DataFrame], col: str, color: str, label: str) -> None:
    """Mean over seeds with a +/- 1 std band. NaN rows are epochs that were not measured."""
    s = pd.concat([f.set_index("epoch")[col] for f in frames], axis=1).dropna(how="all")
    if s.empty:
        return
    mean, std = s.mean(axis=1), s.std(axis=1).fillna(0.0)
    ax.plot(s.index, mean, color=color, label=label)
    ax.fill_between(s.index, mean - std, mean + std, color=color, alpha=0.18, lw=0)


def fig_alignment(cfg: Config, out_dir: Path, n: int = 12) -> None:
    """Overlay the pooled patch grid on the image. One glance catches geometry bugs that no
    assertion will."""
    ds = CachedCocoInstances(cfg, "val")
    n = min(n, len(ds))
    g, p = cfg.grid, cfg.data.patch_size
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12.8, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        s = ds[i]
        im = denormalize(normalize_batch(s["image"][None], torch.device("cpu")))[0]
        grid = s["grid"].numpy().reshape(g, g)
        up = np.repeat(np.repeat(grid, p, 0), p, 1)
        axes[i].imshow(im.permute(1, 2, 0).numpy())
        axes[i].imshow(np.ma.masked_where(up == 0, up), alpha=0.45, cmap="tab20",
                       interpolation="nearest")
        axes[i].set_title(f"coverage {(grid != 0).mean():.2f}, {len(np.unique(grid)) - 1} objects",
                          fontsize=9)
    fig.suptitle("Alignment check: grid cells must sit on the objects")
    _save(fig, out_dir, "F0_alignment_check")


def _load_mask_batch(ds: CachedCocoInstances, n: int) -> dict:
    return {k: torch.stack([ds[i][k] for i in range(n)]) for k in ("image", "grid", "inst_cats")}


def fig_masking(cfg: Config, out_dir: Path, n: int = 5) -> None:
    ds = CachedCocoInstances(cfg, "val")
    batch = _load_mask_batch(ds, n)
    imgs = denormalize(normalize_batch(batch["image"], torch.device("cpu")))
    rng = np.random.default_rng(0)
    g, p = cfg.grid, cfg.data.patch_size
    uniform = np.ones(ds.n_cats, np.float32)

    rows = [("image", None), ("A: random", "random"), ("A2: donor blobs", "shuffled"),
            ("B: own objects", "instance")]
    fig, axes = plt.subplots(len(rows), n, figsize=(2.1 * n, 2.1 * len(rows)))
    for r, (label, strategy) in enumerate(rows):
        masks = None
        if strategy:
            masks, _ = build_batch_masks(batch, strategy, cfg.n_masked, uniform, rng)
        for c in range(n):
            im = imgs[c].permute(1, 2, 0).numpy().copy()
            if masks is not None:
                m = masks[c].numpy().reshape(g, g)
                im[np.repeat(np.repeat(m, p, 0), p, 1)] = 0.5
            axes[r, c].imshow(im)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(label, fontsize=10, color=INK)
    fig.suptitle(f"Every arm hides exactly {cfg.n_masked} of {cfg.num_patches} patches")
    fig.tight_layout()
    _save(fig, out_dir, "F1_masking_examples")


def fig_reconstructions(cfg: Config, runs: Path, out_dir: Path, n: int = 5) -> None:
    from ..engine.evaluate import denorm_pred, load_or_build_bank
    from ..engine.heads import load_encoder
    from . import device as dev

    device = dev.select_device(cfg.device)
    ds = CachedCocoInstances(cfg, "val")
    bank = load_or_build_bank(cfg, ds, cfg.n_masked)
    imgs = normalize_batch(torch.stack([ds[i]["image"] for i in range(n)]), device)
    mask = torch.from_numpy(bank["instance"][:n]).to(device)
    g, p = cfg.grid, cfg.data.patch_size

    panels = [("image", denormalize(imgs).cpu())]
    orig = denormalize(imgs).cpu().permute(0, 2, 3, 1).numpy().copy()
    grey = np.repeat(np.repeat(mask.cpu().numpy().reshape(n, g, g), p, 1), p, 2)
    orig[grey] = 0.5
    panels.append(("masked input", torch.from_numpy(orig).permute(0, 3, 1, 2)))
    for arm in ARMS:
        if not (runs / arm / "seed_0" / "ckpt.pt").exists():
            continue
        model = load_encoder(cfg, arm, 0, device)
        with torch.no_grad():
            _, pred, _ = model(imgs, mask)
            full = model.unpatchify(denorm_pred(model, imgs, pred.float()))
            shown = torch.where(
                mask.reshape(n, 1, g, g).repeat_interleave(p, 2).repeat_interleave(p, 3),
                full, imgs)
        panels.append((ARM_LABEL[arm], denormalize(shown).cpu()))
        del model

    fig, axes = plt.subplots(len(panels), n, figsize=(2.1 * n, 2.1 * len(panels)))
    for r, (label, t) in enumerate(panels):
        for c in range(n):
            axes[r, c].imshow(t[c].permute(1, 2, 0).numpy().clip(0, 1))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(label, fontsize=10, color=INK)
    fig.suptitle("Filling in the same hidden objects (seed 0)")
    fig.tight_layout()
    _save(fig, out_dir, "F2_reconstructions")


def fig_training_curves(runs: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arm in ARMS:
        frames = _logs(runs, arm)
        if frames:
            for ax, col in zip(axes, ("train_loss", "val_loss_random", "val_loss_instance")):
                _band(ax, frames, col, ARM_COLOR[arm], ARM_LABEL[arm])
    for ax, title in zip(axes, ("train loss, each arm's own masks", "val loss, frozen random masks",
                                "val loss, frozen object masks")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("normalised-pixel MSE")
    axes[0].legend()
    fig.suptitle("Only the two frozen-mask panels are comparable across arms", y=1.02)
    _save(fig, out_dir, "F3_training_curves")


def fig_per_category(runs: Path, out_dir: Path, names: list[str], top: int = 20) -> None:
    per_arm = {}
    for arm in ARMS:
        cats = [np.load(p) for p in sorted((runs / arm).glob("seed_*/cat_mse.npy"))]
        if cats:
            per_arm[arm] = np.nanmean(np.stack(cats), axis=0)
    if not per_arm:
        return
    ref = np.nanmean(np.stack(list(per_arm.values())), axis=0)
    order = np.argsort(-np.where(np.isfinite(ref), ref, -np.inf))[:top]

    fig, ax = plt.subplots(figsize=(7.5, 0.3 * top + 1))
    y = np.arange(len(order))
    for arm, values in per_arm.items():
        ax.scatter(values[order], y, s=34, color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, [names[j] for j in order])
    ax.invert_yaxis()
    ax.set_xlabel("reconstruction MSE on frozen object masks (normalised pixels)")
    ax.set_title(f"The {top} categories hardest to reconstruct")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "F4_per_category_mse")


def fig_adaptive_heatmap(runs: Path, out_dir: Path, names: list[str]) -> None:
    """What the adaptive policy learned to prioritise, averaged over seeds."""
    hist = [np.load(p) for p in sorted((runs / "adaptive").glob("seed_*/cat_ema.npy"))]
    if not hist:
        return
    n_epochs = min(h.shape[0] for h in hist)
    weights = np.mean([h[:n_epochs, 1, :] for h in hist], axis=0).T  # (cats, epochs)
    order = np.argsort(-weights[:, -1])

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(weights[order], aspect="auto", cmap=DIVERGING, interpolation="nearest",
                   norm=TwoSlopeNorm(vmin=weights.min(), vcenter=1.0, vmax=weights.max()))
    ax.grid(False)
    ax.set_yticks(range(len(order)), [names[i] for i in order], fontsize=6)
    ax.set_xlabel("epoch")
    ax.set_title("Category sampling weight during training, arm C")
    fig.colorbar(im, ax=ax, label="sampling weight (1 = uniform)", fraction=0.03)
    _save(fig, out_dir, "F5_adaptive_weights")


def fig_category_cv(runs: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in ("instance", "adaptive"):
        frames = _logs(runs, arm)
        if frames:
            _band(ax, frames, "cat_cv", ARM_COLOR[arm], ARM_LABEL[arm])
    ax.set_xlabel("epoch")
    ax.set_ylabel("CV of per-category loss")
    ax.set_title("Does adaptive weighting even out category difficulty?")
    ax.legend()
    _save(fig, out_dir, "F6_category_cv")


def fig_attention_distance(runs: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for key, ax, title in (("attn_dist_full", axes[0], "all 196 tokens"),
                           ("attn_dist_masked", axes[1], "pretraining length (frozen mask)")):
        for arm in ARMS:
            rows = [json.loads(p.read_text("utf-8"))[key]
                    for p in sorted((runs / arm).glob("seed_*/eval.json"))]
            if not rows:
                continue
            a = np.array(rows)
            x = np.arange(1, a.shape[1] + 1)
            ax.plot(x, a.mean(0), color=ARM_COLOR[arm], marker="o", ms=4, label=ARM_LABEL[arm])
            ax.fill_between(x, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=ARM_COLOR[arm],
                            alpha=0.18, lw=0)
        ax.set_xlabel("encoder layer")
        ax.set_title(title)
    axes[0].set_ylabel("mean attention distance (px)")
    axes[0].legend()
    _save(fig, out_dir, "F7_attention_distance")


def fig_coverage(cfg: Config, runs: Path, out_dir: Path) -> None:
    report = Path(cfg.data.cache_dir) / "filter_report.json"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    if report.exists():
        rep = json.loads(report.read_text("utf-8"))
        bins = np.array(rep["coverage_hist"]["bins"])
        axes[0].bar(bins[:-1], rep["coverage_hist"]["counts"], width=np.diff(bins) * 0.92,
                    align="edge", color=ARM_COLOR["random"])
        axes[0].axvline(rep["resolved_cov_min"], color=INK, ls="--",
                        label=f"minimum coverage {rep['resolved_cov_min']:.2f}")
        axes[0].axvline(rep["resolved_mask_ratio"], color=MUTED, ls=":",
                        label=f"mask ratio r = {rep['resolved_mask_ratio']:.2f}")
        axes[0].set_xlabel("share of patches owned by an object")
        axes[0].set_ylabel("images")
        axes[0].set_title("Object coverage across COCO train2017")
        axes[0].legend()
    for arm in ("shuffled", "instance", "adaptive"):
        frames = _logs(runs, arm)
        if frames:
            _band(axes[1], frames, "inst_frac", ARM_COLOR[arm], ARM_LABEL[arm])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("share of the mask made of whole objects")
    axes[1].set_title("The rest of the budget is background top-up")
    axes[1].legend(loc="lower right")
    _save(fig, out_dir, "F8_coverage")


def fig_heads(results: Path, out_dir: Path, split: str) -> None:
    """One small multiple per head: the heads are different metrics on different scales, so
    they do not share an axis. Every seed is a dot and the line is the mean."""
    path = results / f"heads_{split}_per_seed.csv"
    if not path.exists():
        return
    long = pd.read_csv(path)
    heads = [h for h in HEADS if h in set(long["head"])]
    arms = [a for a in ARMS if a in set(long["arm"])]
    cols = 4
    rows = int(np.ceil(len(heads) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel()[len(heads):]:
        ax.axis("off")
    for ax, head in zip(axes.ravel(), heads):
        for i, arm in enumerate(arms):
            v = long[(long["head"] == head) & (long["arm"] == arm)]["value"].to_numpy()
            if v.size == 0:
                continue
            ax.scatter(np.full(v.size, i), v, s=26, color=ARM_COLOR[arm], alpha=0.75,
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.hlines(v.mean(), i - 0.32, i + 0.32, color=ARM_COLOR[arm], lw=2.2, zorder=4)
        ax.set_xticks(range(len(arms)), [ARM_LABEL[a].split(":")[0] for a in arms], rotation=40,
                      ha="right", fontsize=8)
        ax.set_title(HEAD_LABEL[head])
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("macro mAP (mIoU for dense)")
    fig.suptitle(f"Every head, every arm, every seed ({split} set); line = mean of seeds", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "F9_heads")


def fig_contrasts(results: Path, out_dir: Path) -> None:
    """Forest plot: one panel per contrast, one row per head, 95% CI from the bootstrap over
    both seeds and images. Filled means the Holm-adjusted p is below 0.05."""
    path = results / "contrasts.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    heads = [h for h in HEADS if h in set(df["head"])]
    y = np.arange(len(heads))
    fig, axes = plt.subplots(1, len(FAMILY), figsize=(4.0 * len(FAMILY), 0.42 * len(heads) + 1.6),
                             sharey=True, sharex=True)
    for ax, key in zip(axes, FAMILY):
        sub = df[(df["a"] == key[0]) & (df["b"] == key[1])].set_index("head").reindex(heads)
        ax.axvline(0, color=MUTED, lw=1)
        ax.hlines(y, sub["ci_lo"], sub["ci_hi"], color=ARM_COLOR[key[0]], lw=2)
        sig = (sub["p_holm"] < 0.05).to_numpy()
        ax.scatter(sub["diff"][sig], y[sig], s=48, color=ARM_COLOR[key[0]], zorder=3)
        ax.scatter(sub["diff"][~sig], y[~sig], s=48, facecolor="white",
                   edgecolor=ARM_COLOR[key[0]], linewidth=1.6, zorder=3)
        ax.set_title(CONTRAST_LABEL[key])
        ax.set_xlabel("difference in score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [HEAD_LABEL[h] for h in heads])
    axes[0].invert_yaxis()
    fig.suptitle("Filled: significant after Holm correction. Open: not. Bars: 95% CI.",
                 fontsize=10, y=1.03)
    _save(fig, out_dir, "F10_contrasts")


def fig_class_attention(cfg: Config, runs: Path, out_dir: Path, n: int = 5) -> None:
    """Where the class-attention head looks, for the same images and classes in every arm."""
    from ..engine.heads import load_encoder
    from ..models.heads import ClassAttentionHead
    from . import device as dev

    device = dev.select_device(cfg.device)
    ds = CachedCocoInstances(cfg, "val")
    # Images whose largest labelled object is not `person`, so the columns show varied classes.
    picks: list[tuple[int, int]] = []
    for i in range(len(ds)):
        cg = ds[i]["cat_grid"].numpy()
        counts = np.bincount(cg[cg > 0] - 1, minlength=ds.n_cats)
        c = int(counts.argmax())
        if counts[c] >= 12 and ds.cat_names[c] != "person" and c not in [p[1] for p in picks]:
            picks.append((i, c))
        if len(picks) == n:
            break
    imgs = normalize_batch(torch.stack([ds[i]["image"] for i, _ in picks]), device)
    shown = denormalize(imgs).cpu().permute(0, 2, 3, 1).numpy()
    g, p = cfg.grid, cfg.data.patch_size

    arms = [a for a in ARMS if (runs / a / "seed_0" / "class_attn.pt").exists()]
    if not arms:
        return
    fig, axes = plt.subplots(len(arms), n, figsize=(2.1 * n, 2.1 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        state = torch.load(runs / arm / "seed_0" / "class_attn.pt", map_location=device,
                           weights_only=False)
        model = load_encoder(cfg, arm, 0, device)
        head = ClassAttentionHead(state["mu"].numel(), ds.n_cats).to(device)
        head.load_state_dict(state["head"])
        with torch.no_grad():
            tokens = (model.tokens(imgs, cfg.heads.last_k) - state["mu"]) / state["sd"]
            attn = head.attention(tokens).cpu().numpy()  # (n, L, C)
        for c, (_, cat) in enumerate(picks):
            a = attn[c, :, cat].reshape(g, g)
            up = np.repeat(np.repeat(a / a.max(), p, 0), p, 1)
            ax = axes[r, c]
            ax.imshow(shown[c])
            ax.imshow(up, cmap="magma", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(ds.cat_names[cat], fontsize=10)
        axes[r, 0].set_ylabel(ARM_LABEL[arm], fontsize=10, color=INK)
        del model
    fig.suptitle("Where the class-attention head looks (seed 0, brighter = more weight)")
    fig.tight_layout()
    _save(fig, out_dir, "F11_class_attention")


def main() -> None:
    from ..engine.train import resolve_mask_ratio

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--align-check", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args.overrides)
    cfg.mask.ratio = resolve_mask_ratio(cfg)
    runs, results = Path(cfg.train.out_dir), Path(args.results)
    out_dir = results / "figures"

    fig_alignment(cfg, out_dir)
    if args.align_check:
        return

    names = json.loads((Path(cfg.data.cache_dir) / "meta.json").read_text("utf-8"))["cat_names"]
    fig_masking(cfg, out_dir)
    fig_training_curves(runs, out_dir)
    fig_per_category(runs, out_dir, names)
    fig_adaptive_heatmap(runs, out_dir, names)
    fig_category_cv(runs, out_dir)
    fig_attention_distance(runs, out_dir)
    fig_coverage(cfg, runs, out_dir)
    fig_heads(results, out_dir, args.split)
    fig_contrasts(results, out_dir)
    if args.all and seeds_of(runs, "random"):
        fig_reconstructions(cfg, runs, out_dir)
        fig_class_attention(cfg, runs, out_dir)


if __name__ == "__main__":
    main()
