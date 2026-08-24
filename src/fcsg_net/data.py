"""DIV2K patch dataset.

Yields random 128x128 crops from a directory of HR PNGs. Degradation is applied
in the training loop, not here, so the Phase 2 pipeline can slot in unchanged.
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


class PatchDataset(Dataset):
    """Random crops with flip + 90-degree rotation augmentation.

    seed=None gives fresh randomness every epoch (training). An int seed makes
    the crop for a given index identical across runs and machines, which is what
    validation needs for numbers to be comparable between Kaggle sessions.
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
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        p = self.patch
        if w < p or h < p:
            img = img.resize((max(w, p), max(h, p)), Image.BICUBIC)
            w, h = img.size
        x, y = rng.randint(0, w - p), rng.randint(0, h - p)
        img = img.crop((x, y, x + p, y + p))

        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)

        if rng.random() < 0.5:
            t = torch.flip(t, dims=[2])
        k = rng.randint(0, 3)
        if k:
            t = torch.rot90(t, k, dims=[1, 2])
        return {"hr": t.contiguous(), "path": str(self.files[idx])}


def main():
    ap = argparse.ArgumentParser(description="render a crop grid to eyeball the pipeline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="data_check.png")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    from torchvision.utils import save_image

    ds = PatchDataset(args.data, patch=args.patch, seed=0)
    print(f"{len(ds)} images under {args.data}")
    batch = torch.stack([ds[i]["hr"] for i in range(args.n)])
    assert batch.shape == (args.n, 3, args.patch, args.patch), batch.shape
    assert 0.0 <= batch.min() and batch.max() <= 1.0
    # a grid of identical or constant crops means the crop/augment logic is broken
    assert batch.std() > 0.02, "crops are near-constant, check decoding"
    save_image(batch, args.out, nrow=4)
    print(f"wrote {args.out}  mean={batch.mean():.3f} std={batch.std():.3f}")


if __name__ == "__main__":
    main()
