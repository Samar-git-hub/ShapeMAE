"""The small amount of GPU-specific code: device choice, bf16 autocast, keeping awake."""
from __future__ import annotations

import ctypes
import sys
from contextlib import nullcontext

import torch


def select_device(prefer: str = "auto") -> torch.device:
    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda:0")
    if prefer == "cuda":
        raise SystemExit("device=cuda requested but torch cannot see a GPU")
    return torch.device("cpu")


def autocast(device: torch.device, enabled: bool = True):
    """bf16 needs no GradScaler. On CPU (the unit tests) it is a no-op."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def peak_vram_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0


def keep_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs. Lapses when the process exits,
    so no power setting is changed."""
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
