"""Pre-extract DIV2K crops into a uint8 .npy memmap.

Decoding a 2K PNG for every training sample caps the loader at roughly
1.8 steps/s, which is a 46-hour run at 300k steps. Tiles come off a memmap
instead, so the bottleneck moves back to the GPU. Run once per patch size.

    python training/build_tiles.py --data /kaggle/input --out /kaggle/working/tiles.npy
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fcsg_net.data import ensure_min_size, find_images  # noqa: E402
from fcsg_net.utils import resolve_div2k  # noqa: E402


def build(files, out, patch, per_image, seed=0):
    rng = random.Random(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(files) * per_image
    tiles = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.uint8, shape=(n, patch, patch, 3)
    )
    print(f"{len(files)} images x {per_image} crops = {n} tiles "
          f"({tiles.nbytes / 1e9:.2f} GB) -> {out}")

    t0 = time.time()
    for i, f in enumerate(files):
        img = ensure_min_size(Image.open(f).convert("RGB"), patch)
        a = np.asarray(img)
        h, w = a.shape[:2]
        for j in range(per_image):
            y, x = rng.randint(0, h - patch), rng.randint(0, w - patch)
            tiles[i * per_image + j] = a[y : y + patch, x : x + patch]
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f"  {i + 1}/{len(files)} images  {time.time() - t0:.0f}s")
    tiles.flush()
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="any ancestor of the DIV2K image directories")
    ap.add_argument("--train-dir", help="skip the search and use this directory")
    ap.add_argument("--val-dir")
    ap.add_argument("--out", required=True, help="path to the .npy to write")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--per-image", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_dir, _ = resolve_div2k(args.data, args.train_dir, args.val_dir)
    files = find_images(train_dir)
    print(f"train={train_dir}")
    tiles = build(files, args.out, args.patch, args.per_image, args.seed)

    sample = np.asarray(tiles[:: max(len(tiles) // 256, 1)], dtype=np.float32)
    assert sample.std() > 5.0, "tiles are near-constant, check decoding"
    print(f"done  mean={sample.mean():.1f}  std={sample.std():.1f}")


if __name__ == "__main__":
    main()
