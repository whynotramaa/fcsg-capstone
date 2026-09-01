"""DIV2K patch datasets.

PatchDataset crops straight from PNGs. TileDataset reads a pre-extracted uint8
memmap built by training/build_tiles.py, which is what real runs should use:
decoding a 2K PNG per sample caps the loader at roughly 1.8 steps/s.

Degradation is applied in the training loop, not here, so the Phase 2 pipeline
slots in unchanged.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images(root):
    """All images under root, sorted. Recurses, so a dataset dir or its parent works."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images under {root}")
    return files


def _augment(t, rng):
    if rng.random() < 0.5:
        t = torch.flip(t, dims=[2])
    k = rng.randint(0, 3)
    if k:
        t = torch.rot90(t, k, dims=[1, 2])
    return t.contiguous()


def _to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255.0)


def ensure_min_size(img, p):
    """Upscale proportionally if either side is under p, so crops never distort."""
    w, h = img.size
    if w < p or h < p:
        s = p / min(w, h)
        img = img.resize((max(p, round(w * s)), max(p, round(h * s))), Image.BICUBIC)
    return img


class PatchDataset(Dataset):
    """Random crops with flip and 90-degree rotation augmentation.

    seed=None gives fresh randomness every epoch (training). An int seed makes
    the crop for a given index identical across runs and machines, which is what
    validation needs for numbers to be comparable between sessions.
    """

    def __init__(self, root, patch=128, seed=None, limit=None):
        self.files = find_images(root)
        if limit is not None:
            self.files = self.files[:limit]
        self.patch = patch
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx) if self.seed is not None else random
        p = self.patch
        img = ensure_min_size(Image.open(self.files[idx]).convert("RGB"), p)
        w, h = img.size
        x, y = rng.randint(0, w - p), rng.randint(0, h - p)
        t = _to_tensor(np.array(img.crop((x, y, x + p, y + p))))
        return {"hr": _augment(t, rng)}


class TileDataset(Dataset):
    """Pre-extracted tiles from a .npy memmap, shape (N, patch, patch, 3) uint8."""

    def __init__(self, path, seed=None):
        self.path = Path(path)
        self.seed = seed
        self.tiles = None
        self.n = len(np.load(self.path, mmap_mode="r"))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.tiles is None:
            # Opened on first access so every DataLoader worker maps its own handle.
            self.tiles = np.load(self.path, mmap_mode="r")
        rng = random.Random(self.seed + idx) if self.seed is not None else random
        return {"hr": _augment(_to_tensor(self.tiles[idx]), rng)}


def main():
    ap = argparse.ArgumentParser(description="render a crop grid to eyeball the pipeline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data_check.png")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tiles", action="store_true", help="--data points at a .npy tile cache")
    args = ap.parse_args()

    from torchvision.utils import save_image

    ds = TileDataset(args.data, seed=0) if args.tiles else PatchDataset(args.data, patch=args.patch, seed=0)
    print(f"{len(ds)} samples in {args.data}")
    batch = torch.stack([ds[i]["hr"] for i in range(args.n)])
    assert batch.shape[0] == args.n and batch.shape[1] == 3, batch.shape
    assert 0.0 <= batch.min() and batch.max() <= 1.0
    assert batch.std() > 0.02, "crops are near-constant, check decoding"
    save_image(batch, args.out, nrow=4)
    print(f"wrote {args.out}  mean={batch.mean():.3f} std={batch.std():.3f}")


if __name__ == "__main__":
    main()
