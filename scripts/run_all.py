"""Run the whole experiment. Safe to stop and restart at any point: finished stages are
skipped and an interrupted pretraining run resumes from its last checkpoint.

    python scripts/run_all.py pretrain
    python scripts/run_all.py pretrain --seeds 0
    python scripts/run_all.py heads --split test

Seeds are the outer loop, so if the machine is needed back early, what exists is a balanced
set of complete seeds rather than some arms with more runs than others.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMS = {"random": "arm_a_random", "shuffled": "arm_a2_shuffled", "instance": "arm_b_instance",
        "adaptive": "arm_c_adaptive"}
RETRIES = 3


def run(module: str, *args: str) -> None:
    """Each stage gets its own process, so a crash cannot take the others down with it."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n=== {' '.join(cmd[2:])}", flush=True)
    for attempt in range(RETRIES):
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"[retry {attempt + 1}/{RETRIES}] {module} failed", flush=True)
    raise SystemExit(f"giving up on: {' '.join(cmd)}")


def pretrain(seeds: list[int], extra: list[str]) -> None:
    for seed in seeds:
        for arm, config in ARMS.items():
            run("src.engine.train", "--config", f"configs/{config}.yaml", "--seed", str(seed),
                "--resume", *extra)
            if not (Path("runs") / arm / f"seed_{seed}" / "eval.json").exists():
                run("src.engine.evaluate", "--arm", arm, "--seed", str(seed), *extra)


def has(arm: str, seed: int, split: str, key: str) -> bool:
    path = Path("runs") / arm / f"seed_{seed}" / f"heads_{split}.json"
    return path.exists() and key in json.loads(path.read_text("utf-8"))


def heads(seeds: list[int], split: str, extra: list[str]) -> None:
    for seed in seeds:
        for arm in ARMS:
            common = ("--arm", arm, "--seed", str(seed), "--split", split, *extra)
            if not has(arm, seed, split, "dense_miou"):
                run("src.engine.heads", *common)
            if not has(arm, seed, split, "finetune"):
                run("src.engine.finetune", *common)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["pretrain", "heads"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()
    extra = ["--set", *args.overrides] if args.overrides else []

    if args.stage == "pretrain":
        pretrain(args.seeds, extra)
    else:
        heads(args.seeds, args.split, extra)


if __name__ == "__main__":
    main()
