"""On-the-fly archival degradation and a reproducible D2 snapshot."""

import argparse
import csv
import io
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


def degrade(hr, rng, resample=True):
    """Return an RGB tensor and the sampled parameters. Input is CPU CHW [0, 1]."""
    if hr.ndim != 3 or hr.shape[0] != 3 or min(hr.shape[-2:]) < 2:
        raise ValueError("expected an RGB CHW image at least 2x2")
    if hr.device.type != "cpu" or not torch.isfinite(hr).all() or hr.min() < 0 or hr.max() > 1:
        raise ValueError("degradation expects a finite CPU image in [0, 1]")
    params = {"blur_sigma": float(rng.uniform(0.5, 2.0)),
              "scale": float(rng.uniform(1, 2)) if resample else 1.0,
              "noise_sigma": float(rng.uniform(5, 50)),
              "jpeg_quality": int(rng.integers(30, 96))}
    a = hr.permute(1, 2, 0).numpy()
    img = Image.fromarray(np.rint(a * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(params["blur_sigma"]))
    if resample:
        size = img.size
        small = tuple(max(1, round(s / params["scale"])) for s in size)
        img = img.resize(small, Image.Resampling.BICUBIC).resize(size, Image.Resampling.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255
    a = np.clip(a + rng.normal(0, params["noise_sigma"] / 255, a.shape), 0, 1)
    buffer = io.BytesIO()
    Image.fromarray(np.rint(a * 255).astype(np.uint8)).save(
        buffer, format="JPEG", quality=params["jpeg_quality"], subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        a = np.array(decoded.convert("RGB"), copy=True)
    return torch.from_numpy(a).permute(2, 0, 1).float() / 255, params


class DegradedDataset(Dataset):
    def __init__(self, clean, seed=None, resample=True):
        self.clean, self.seed, self.resample = clean, seed, resample

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        sample = self.clean[idx]
        # DataLoader seeds Python random per worker; an explicit seed fixes validation.
        seed = self.seed + idx if self.seed is not None else random.getrandbits(63)
        lr, params = degrade(sample["hr"], np.random.default_rng(seed), self.resample)
        return {**sample, "lr": lr, **params}


def snapshot(data, out, count=5000, seed=1234, patch=128):
    from fcsg_net.data import PatchDataset

    if count < 1 or patch < 2:
        raise ValueError("count must be positive and patch >= 2")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "hr").mkdir()
    (out / "lr").mkdir()
    clean = PatchDataset(data, patch=patch, seed=seed)
    t0 = time.perf_counter()
    with (out / "manifest.csv").open("w", newline="") as f:
        fields = ["id", "source", "crop_seed", "noise_seed", "blur_sigma", "scale", "noise_sigma", "jpeg_quality"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(count):
            # Fresh deterministic crop even after cycling through the source images.
            clean.seed = seed + i * len(clean)
            hr = clean[i % len(clean)]["hr"]
            lr, params = degrade(hr, np.random.default_rng(seed + i))
            for name, tensor in (("hr", hr), ("lr", lr)):
                a = tensor.permute(1, 2, 0).numpy()
                Image.fromarray(np.rint(a * 255).astype(np.uint8)).save(out / name / f"{i:05d}.png")
            writer.writerow({"id": i, "source": str(clean.files[i % len(clean)].relative_to(data)),
                             "crop_seed": clean.seed + i % len(clean), "noise_seed": seed + i, **params})
            if (i + 1) % 250 == 0:
                print(f"D2 snapshot {i + 1}/{count}", flush=True)
    return {"pairs": count, "seed": seed, "patch": patch,
            "seconds": time.perf_counter() - t0, "path": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training HR images only")
    ap.add_argument("--out", required=True, help="new directory for 5,000 PNG pairs and metadata")
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    print(snapshot(args.data, args.out, args.count, args.seed))
