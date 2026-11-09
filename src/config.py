"""Typed configuration: dataclasses + YAML with `_base` inheritance and `--set` overrides.

Every run writes its fully resolved config next to its checkpoint, which makes the cross-arm
comparison auditable: the arms must differ in `mask.*` and `adaptive.enabled`, nothing else.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    cache_dir: str = "data/cache"
    test_cache_dir: str = "data/cache_test"
    coco_ann: str = "data/coco/annotations/instances_train2017.json"
    coco_ann_test: str = "data/coco/annotations/instances_val2017.json"
    coco_images: str = "data/coco"  # images live at <coco_images>/<split>/<id>.jpg

    cache_size: int = 256  # cached square side; RandomResizedCrop draws from this
    img_size: int = 224
    patch_size: int = 16  # -> 14x14 grid, L = 196

    subset_seed: int = 51  # which curated images are drawn; separate from the training seed
    n_train: int = 13500
    n_val: int = 1500
    max_instances: int = 40
    patch_assign_thresh: float = 0.25  # an instance owns a patch at >= 25% occupancy

    # structural curation filter, all measured on the patch grid
    min_instances: int = 2
    min_instance_patches: int = 2
    cov_max: float = 0.95
    largest_instance_max: float = 0.70
    min_side_px: int = 200

    rrc_scale: tuple[float, float] = (0.5, 1.0)
    rrc_ratio: tuple[float, float] = (0.75, 1.3333)
    hflip: bool = True

    num_workers: int = 4


@dataclass
class ModelCfg:
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    decoder_embed_dim: int = 128
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    mlp_ratio: float = 4.0
    norm_pix_loss: bool = True


@dataclass
class MaskCfg:
    strategy: str = "random"  # random | shuffled | instance | adaptive
    ratio: float = 0.40  # fallback only; the ratio resolved by curation wins
    ratio_override: float | None = None


@dataclass
class AdaptiveCfg:
    enabled: bool = False
    momentum: float = 0.90  # EPOCH-level EMA -> horizon of about 10 epochs
    warmup_epochs: int = 40
    tau_start: float = 0.0
    tau_end: float = 4.0
    w_min: float = 0.25
    w_max: float = 4.0
    min_patches_per_cat: int = 32  # ignore a category's epoch statistic below this many patches
    loss_clip_q: float = 0.99  # winsorise per-patch loss before the difficulty estimate


@dataclass
class TrainCfg:
    epochs: int = 400
    batch_size: int = 128
    base_lr: float = 2.4e-3  # effective lr = base_lr * batch_size / 256
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 40
    min_lr: float = 0.0
    amp: bool = True  # bf16; Ada has native support, so no GradScaler is needed
    grad_clip: float = 3.0
    log_every: int = 20
    eval_every: int = 5
    ckpt_every: int = 10  # resume point; a laptop can always reboot mid-run
    out_dir: str = "runs"


@dataclass
class EvalCfg:
    mask_bank: str = "data/cache/eval_masks.npz"
    bank_seed: int = 12345
    knn_k: int = 20
    bootstrap: int = 1000


@dataclass
class HeadsCfg:
    last_k: int = 4  # frozen features concatenate the last k encoder blocks
    batch_size: int = 256
    linear_epochs: int = 100
    linear_lr: float = 3.0e-2
    attn_epochs: int = 60
    attn_what_lr: float = 1.0e-2
    attn_where_lr: float = 3.0e-4
    dense_epochs: int = 30
    dense_lr: float = 0.3
    weight_decay: float = 1.0e-4

    ft_epochs: int = 50
    ft_batch_size: int = 128
    ft_lr: float = 3.0e-3
    ft_warmup_epochs: int = 5
    ft_weight_decay: float = 0.05


@dataclass
class Config:
    seed: int = 0
    arm: str = "random"
    device: str = "auto"  # auto | cuda | cpu
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    adaptive: AdaptiveCfg = field(default_factory=AdaptiveCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    heads: HeadsCfg = field(default_factory=HeadsCfg)

    @property
    def grid(self) -> int:
        return self.data.img_size // self.data.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid**2

    @property
    def n_masked(self) -> int:
        """M, the masking budget. Identical for every arm, by construction."""
        return round(self.mask.ratio * self.num_patches)

    @property
    def run_dir(self) -> Path:
        return Path(self.train.out_dir) / self.arm / f"seed_{self.seed}"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dataclasses.asdict(self), sort_keys=False), "utf-8")


_SECTIONS = {"data": DataCfg, "model": ModelCfg, "mask": MaskCfg, "adaptive": AdaptiveCfg,
             "train": TrainCfg, "eval": EvalCfg, "heads": HeadsCfg}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(cls, d: dict):
    """Build a dataclass from a dict, tolerating missing keys and fixing tuple fields."""
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in d:
            kwargs[f.name] = tuple(d[f.name]) if isinstance(d[f.name], list) else d[f.name]
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML config. A config may declare `_base: other.yaml` to inherit."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text("utf-8")) or {}
    if "_base" in raw:
        base_path = (path.parent / raw.pop("_base")).resolve()
        raw = _deep_merge(yaml.safe_load(base_path.read_text("utf-8")) or {}, raw)

    cfg = Config(**{k: raw[k] for k in ("seed", "arm", "device") if k in raw})
    for name, cls in _SECTIONS.items():
        if raw.get(name):
            setattr(cfg, name, _coerce(cls, raw[name]))
    return cfg


def apply_cli_overrides(cfg: Config, pairs: list[str]) -> Config:
    """`--set train.epochs=50 mask.ratio=0.6` -> typed assignment onto the config."""
    for pair in pairs or []:
        key, _, val = pair.partition("=")
        section, _, leaf = key.partition(".")
        target = getattr(cfg, section) if leaf else cfg
        name = leaf or section
        if not hasattr(target, name):
            raise KeyError(f"unknown config key: {key}")
        cur = getattr(target, name)
        new: Any
        if isinstance(cur, bool):
            new = val.lower() in {"1", "true", "yes"}
        elif isinstance(cur, int):
            new = int(float(val))
        elif isinstance(cur, float) or cur is None:
            new = float(val)
        else:
            new = val
        setattr(target, name, new)
    return cfg
