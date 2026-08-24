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


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_dirs(root, limit=50000):
    """Every directory under root that directly holds images, with its count."""
    found = {}
    for dirpath, _, filenames in os.walk(root):
        n = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if n:
            found[Path(dirpath)] = n
        if len(found) >= limit:
            break
    return found


def _pick(dirs, keywords, taken=None):
    """First directory whose name matches a keyword, keywords in priority order."""
    for kw in keywords:
        for d in sorted(dirs):
            if d in (taken or ()):
                continue
            if kw in d.name.lower():
                return d
    return None


def resolve_div2k(root, train_dir=None, val_dir=None):
    """Find the training and validation image directories under root.

    Kaggle dataset layouts differ from each other and from a Colab unzip, so
    nothing hardcodes a path: point --data at /kaggle/input and let it search.
    Pass train_dir and val_dir explicitly to skip the search entirely.
    """
    if train_dir and val_dir:
        return Path(train_dir), Path(val_dir)

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    dirs = image_dirs(root)
    if not dirs:
        raise FileNotFoundError(
            f"no images anywhere under {root}. Attach a DIV2K dataset in the "
            f"Kaggle sidebar before running this."
        )

    train = Path(train_dir) if train_dir else _pick(dirs, ("train_hr", "train"))
    val = Path(val_dir) if val_dir else _pick(dirs, ("valid_hr", "valid", "val", "test"), taken={train})

    if train is None and val is not None:
        train = _pick(dirs, ("hr", ""), taken={val})
    if train is not None and val is None and len(dirs) == 1:
        # One directory, no split. Fine for a smoke run, wrong for a real one:
        # validation crops are drawn from images the model trains on.
        val = train
        print(f"WARNING: only one image directory found ({train}). Validating on "
              f"training images. Attach a dataset with a validation split before "
              f"you trust any PSNR number.")

    if train is None or val is None:
        listing = "\n".join(f"  {d}  ({n} images)" for d, n in sorted(dirs.items())[:20])
        raise FileNotFoundError(
            f"could not identify train and validation directories under {root}.\n"
            f"Directories holding images:\n{listing}\n"
            f"Pass --train-dir and --val-dir explicitly with two of these paths."
        )
    return train, val
