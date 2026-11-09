"""End-to-end check on a synthetic cache: no COCO needed, a few minutes.

    python scripts/smoke_test.py

Exercises every stage, including a mid-run resume, and fails if an expected output is missing,
so a stage that silently does nothing cannot pass.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CACHE, RUNS, RESULTS = "data/smoke", "runs_smoke", "results_smoke"
OVERRIDES = [
    f"data.cache_dir={CACHE}/train", f"data.test_cache_dir={CACHE}/test", "data.cache_size=128",
    "data.img_size=112", "data.num_workers=0", "data.max_instances=12", "model.depth=4",
    "model.decoder_depth=2", "train.epochs=4", "train.batch_size=16", "train.warmup_epochs=1",
    "train.ckpt_every=1", f"train.out_dir={RUNS}", "adaptive.warmup_epochs=0",
    "adaptive.min_patches_per_cat=4", f"eval.mask_bank={CACHE}/eval_masks.npz",
    "eval.bootstrap=50", "heads.linear_epochs=3", "heads.attn_epochs=3",
    "heads.dense_epochs=2", "heads.ft_epochs=2", "heads.ft_warmup_epochs=1",
    "heads.batch_size=16", "heads.ft_batch_size=16",
]
ARMS = {"random": "arm_a_random", "shuffled": "arm_a2_shuffled", "instance": "arm_b_instance",
        "adaptive": "arm_c_adaptive"}
SEEDS = ("0", "1")


def run(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args, "--set", *OVERRIDES], check=True)


def main() -> None:
    for path in (CACHE, RUNS, RESULTS):
        shutil.rmtree(path, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "src.data.synthetic", "--out", CACHE], check=True)

    # Stop one run halfway, as a reboot would. The loop below has to resume it.
    run("src.engine.train", "--config", "configs/arm_a_random.yaml", "--seed", "0",
        "--stop-after", "2")
    if not (Path(RUNS) / "random" / "seed_0" / "resume.pt").exists():
        raise SystemExit("SMOKE TEST FAILED: no resume point after an interrupted run")

    for seed in SEEDS:
        for arm, config in ARMS.items():
            run("src.engine.train", "--config", f"configs/{config}.yaml", "--seed", seed,
                "--resume")
            run("src.engine.evaluate", "--arm", arm, "--seed", seed)
        for arm in ARMS:
            run("src.engine.heads", "--arm", arm, "--seed", seed, "--split", "test")
            run("src.engine.finetune", "--arm", arm, "--seed", seed, "--split", "test")

    run("src.engine.evaluate", "--treatment", "--results", RESULTS)
    run("src.engine.analyze", "--split", "test", "--results", RESULTS)
    run("src.utils.viz", "--all", "--split", "test", "--results", RESULTS)

    expected = [f"{RESULTS}/contrasts.csv", f"{RESULTS}/heads_test.csv",
                f"{RESULTS}/treatment.csv", f"{RESULTS}/reconstruction.csv",
                f"{RESULTS}/figures/F9_heads.png", f"{RESULTS}/figures/F10_contrasts.png",
                f"{RESULTS}/figures/F11_class_attention.png",
                f"{RUNS}/adaptive/seed_1/ckpt.pt", f"{RUNS}/random/seed_0/heads_test.json"]
    missing = [p for p in expected if not Path(p).exists()]
    if missing:
        raise SystemExit(f"SMOKE TEST FAILED, missing: {missing}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
