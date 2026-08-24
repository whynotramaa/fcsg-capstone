"""Shared helpers: degradation for Phase 1, checkpointing, CSV logging."""

import csv
import os
import re
from pathlib import Path

import torch


def add_noise(x, sigma, generator=None):
    """Additive white Gaussian noise. sigma is in [0, 1] units, i.e. 25/255."""
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (x + noise * sigma).clamp(0, 1)


def save_checkpoint(path, step, model, optimizer=None, scaler=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scaler": scaler.state_dict() if scaler else None,
            "extra": extra or {},
        },
        path,
    )


def newest_checkpoint(out_dir):
    """Highest-step ckpt_*.pt in out_dir, or None. Sorts numerically, not lexically."""
    ckpts = list(Path(out_dir).glob("ckpt_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    """Restore state and return the step to resume from."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck["step"]


class CSVLog:
    """Append-only CSV. stdout does not survive a killed Kaggle session; this does."""

    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(fields)

    def write(self, **row):
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def resolve_div2k(root):
    """Find (train_HR, valid_HR) under root.

    Kaggle dataset slugs vary and Colab unzips somewhere else again, so nothing
    hardcodes a path: point --data at /kaggle/input (or any ancestor) and let it
    search.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    train = val = None
    # os.walk, not rglob("*"): /kaggle/input can hold tens of thousands of files
    # and this runs at the top of every notebook.
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            low = d.lower()
            if train is None and "train_hr" in low:
                train = Path(dirpath) / d
            if val is None and "valid_hr" in low:
                val = Path(dirpath) / d
        if train and val:
            break
    if train is None:
        raise FileNotFoundError(f"no *train_HR* directory under {root}")
    if val is None:
        raise FileNotFoundError(f"no *valid_HR* directory under {root}")
    return train, val
