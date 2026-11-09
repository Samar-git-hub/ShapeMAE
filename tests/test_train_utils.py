"""Things a multi-day run on a laptop depends on: logs and configs survive a restart."""
from __future__ import annotations

import csv
from pathlib import Path

from src.config import apply_cli_overrides, load_config
from src.utils.logging import RunLogger

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_resume_keeps_logged_epochs(tmp_path):
    """Regression: restarting used to rewrite log.csv from an empty header."""
    log = RunLogger(tmp_path)
    log.log_epoch({"epoch": 0, "loss": 1.0})
    log.log_epoch({"epoch": 1, "loss": 0.9})

    log = RunLogger(tmp_path, resume=True)
    log.log_epoch({"epoch": 2, "loss": 0.8})

    with (tmp_path / "log.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]


def test_resume_redoes_epochs_after_the_last_checkpoint(tmp_path):
    """A crash at epoch 3 with a checkpoint at epoch 1 must not leave epochs 2 and 3 twice."""
    log = RunLogger(tmp_path)
    for epoch in range(4):
        log.log_epoch({"epoch": epoch, "loss": 1.0})

    log = RunLogger(tmp_path, resume=True)
    log.drop_from(2)
    for epoch in (2, 3):
        log.log_epoch({"epoch": epoch, "loss": 0.5})

    with (tmp_path / "log.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["epoch"] for r in rows] == ["0", "1", "2", "3"]
    assert float(rows[3]["loss"]) == 0.5


def test_fresh_run_starts_a_new_log(tmp_path):
    RunLogger(tmp_path).log_epoch({"epoch": 0, "loss": 1.0})
    RunLogger(tmp_path).log_epoch({"epoch": 0, "loss": 2.0})
    with (tmp_path / "log.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and float(rows[0]["loss"]) == 2.0


def test_arm_configs_differ_only_in_masking():
    """The controlled comparison, as a test: arm files may not change anything else."""
    arms = [load_config(p) for p in sorted(CONFIGS.glob("arm_*.yaml"))]
    assert len(arms) >= 4
    for cfg in arms:
        for section in ("data", "model", "train", "eval", "heads"):
            assert getattr(cfg, section) == getattr(arms[0], section), section


def test_cli_overrides_are_typed():
    cfg = load_config(CONFIGS / "base.yaml")
    apply_cli_overrides(cfg, ["train.epochs=7", "train.base_lr=0.01", "train.amp=false",
                              "mask.ratio_override=0.75"])
    assert cfg.train.epochs == 7 and cfg.train.base_lr == 0.01
    assert cfg.train.amp is False and cfg.mask.ratio_override == 0.75
