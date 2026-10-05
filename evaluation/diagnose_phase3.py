"""Inspect finished Phase 3 checkpoints on Kaggle, without training."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evaluation.eval import load_image, tiled_forward
from fcsg_net.data import find_images
from fcsg_net.degrade import degrade
from fcsg_net.metrics import psnr
from fcsg_net.utils import CSVLog, newest_checkpoint, resolve_div2k
from training.train import build_model, fixed_sigma, predict

METHODS = ("fcsg", "dncnn_composite", "ffdnet_blind", "ffdnet_composite")
PARAMS = ("blur_sigma", "scale", "noise_sigma", "jpeg_quality")


def routing_stats(sparse, dense):
    sparse, dense = sparse.float(), dense.float()
    selected = sparse > 0
    excluded = sparse.argmin(2)
    return {
        "dense_entropy": (-(dense.clamp_min(1e-8).log() * dense).sum(2).mean((0, 2, 3))).tolist(),
        "sparse_entropy": (-(sparse.clamp_min(1e-8).log() * sparse).sum(2).mean((0, 2, 3))).tolist(),
        "mean_weight": sparse.mean((0, 3, 4)).tolist(),
        "selection_fraction": selected.float().mean((0, 3, 4)).tolist(),
        "spatial_std": sparse.flatten(3).std(3, unbiased=False).mean(0).tolist(),
        "dense_spatial_std": dense.flatten(3).std(3, unbiased=False).mean(0).tolist(),
        "pair_fraction": torch.stack([(excluded == e).float().mean((0, 2, 3)) for e in range(3)], 1).tolist(),
    }


def correlations(rows):
    # Center within each image to remove differences in image content.
    groups = sorted({r["image"] for r in rows})
    result = []
    for b in range(3):
        for e in range(3):
            for param in PARAMS:
                xs, ys = [], []
                for image in groups:
                    group = [r for r in rows if r["image"] == image]
                    x = np.array([r[param] for r in group])
                    y = np.array([r["mean_weight"][b][e] for r in group])
                    xs.extend(x - x.mean())
                    ys.extend(y - y.mean())
                x, y = np.array(xs), np.array(ys)
                norm = np.linalg.norm(x) * np.linalg.norm(y)
                result.append({"band": b, "expert": e, "parameter": param,
                               "r": float(x @ y / norm) if norm > 1e-12 else None})
    return result


def self_check():
    dense = torch.full((1, 3, 3, 2, 2), 1 / 3)
    sparse = torch.zeros_like(dense)
    sparse[:, :, :2] = 0.5
    stats = routing_stats(sparse, dense)
    assert np.allclose(stats["dense_entropy"], np.log(3))
    assert np.allclose(stats["sparse_entropy"], np.log(2))
    assert stats["selection_fraction"][0] == [1, 1, 0]
    assert stats["pair_fraction"][0] == [0, 0, 1]
    sparse[:, :, 1, :, 1] = 0
    sparse[:, :, 2, :, 1] = 0.5
    stats = routing_stats(sparse, dense)
    assert stats["pair_fraction"][0] == [0, 0.5, 0.5]
    assert stats["spatial_std"][0][1] == 0.25
    rows = [{"image": image, **{p: offset + j for p in PARAMS},
             "mean_weight": [[offset + j] * 3] * 3}
            for image, offset in (("a", 0), ("b", 100)) for j in range(3)]
    assert all(abs(r["r"] - 1) < 1e-12 for r in correlations(rows))
    for row in rows:
        row["mean_weight"] = [[0.5] * 3] * 3
    assert all(r["r"] is None for r in correlations(rows))
    print("Routing statistics and within-image correlation checks passed.", flush=True)


def latency(model, size, sigma, warmup=20, repeats=50):
    x = torch.rand(1, 3, size, size, device="cuda")
    for _ in range(warmup):
        predict(model, x, sigma)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu, wall = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        t = time.perf_counter()
        start.record()
        predict(model, x, sigma)
        end.record()
        end.synchronize()
        wall.append(1000 * (time.perf_counter() - t))
        gpu.append(start.elapsed_time(end))
    return {"size": size, "gpu_ms_median": float(np.median(gpu)),
            "wall_ms_median": float(np.median(wall)), "wall_ms_p90": float(np.percentile(wall, 90)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "warmup": warmup, "repeats": repeats}


def plot_routing(image, sparse, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for b in range(3):
        axes[b, 0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
        axes[b, 0].set_title(("Low", "Mid", "High")[b] + "-band routing, degraded input")
        for e in range(3):
            axes[b, e + 1].imshow(sparse[0, b, e].cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
            axes[b, e + 1].set_title(f"Expert {e}, kernel {(7, 5, 3)[e]}")
        for ax in axes[b]:
            ax.axis("off")
    fig.suptitle("Selected routing weights, fixed scale 0 to 1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=10)
    ap.add_argument("--routing-samples", type=int, default=8)
    args = ap.parse_args()
    if args.images < 2 or args.routing_samples < 2:
        raise ValueError("need at least two images and routing samples")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic on a Kaggle/Colab GPU")
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    self_check()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    _, val = resolve_div2k(args.input)
    all_files = find_images(val)
    if len(all_files) != 100 or args.images > len(all_files):
        raise ValueError("expected 100 DIV2K validation images and a valid subset size")
    indices = np.linspace(0, len(all_files) - 1, args.images, dtype=int).tolist()
    summary = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "precision": "float32, TF32 disabled",
               "batch": 1, "tile_overlap": 32, "indices": indices,
               "latency_note": "warm forward pass only; excludes image transfers and degradation",
               "routing_note": "256-pixel center crops of whole-image degradations; exploratory correlations, all parameters vary",
               "intervention_note": "inference-only substitutions with the same trained weights, not retrained A2",
               "models": {}}
    tiles = CSVLog(out / "tiles.csv", ["method", "image", "seed", *PARAMS, "tile", "psnr_in", "psnr_out", "seconds"])
    interventions = CSVLog(out / "interventions.csv", ["image", "variant", "psnr_out"])
    routing = []
    for method in METHODS:
        candidates = []
        for status_path in Path(args.input).rglob("status.json"):
            status = json.loads(status_path.read_text())
            if status.get("config") == method and status.get("finished") and status.get("step") == 200000:
                checkpoint = newest_checkpoint(status_path.parent / "ckpt")
                if checkpoint:
                    candidates.append(checkpoint)
        if len(candidates) != 1:
            raise ValueError(f"expected one finished checkpoint for {method}, found {candidates}")
        checkpoint = candidates[0]
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["extra"]["config"]
        assert ck["step"] == 200000 and cfg["degradation"] == "composite"
        assert cfg["model"] == ("ffdnet" if method.startswith("ffdnet") else method.split("_")[0])
        assert (fixed_sigma(cfg) == 27.5 / 255) if method == "ffdnet_blind" else fixed_sigma(cfg) is None
        model = build_model(cfg["model"], **cfg.get("model_args", {})).cuda().eval()
        model.load_state_dict(ck["model"])
        del ck
        sigma = fixed_sigma(cfg)
        level = 27.5 / 255 if sigma is None else sigma
        result = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "latency": [latency(model, size, level) for size in (128, 256, 512)], "tile_psnr": {}}
        scores = {size: [] for size in (128, 256, 512)}
        for idx in indices:
            path = all_files[idx]
            hr = load_image(path, "cpu")
            lr, params = degrade(hr[0], np.random.default_rng(idx))
            noisy, target = lr[None].cuda(), hr.cuda()
            level = params["noise_sigma"] / 255 if sigma is None else sigma
            for size in scores:
                torch.cuda.synchronize()
                t = time.perf_counter()
                pred = tiled_forward(model, noisy, tile=size, overlap=32, sigma=level).clamp(0, 1)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t
                score = psnr(pred, target)
                scores[size].append(score)
                tiles.write(method=method, image=path.name, seed=idx, **params, tile=size,
                            psnr_in=psnr(noisy, target), psnr_out=score, seconds=seconds)
                del pred
            if method == "fcsg":
                for variant in ("equal_selected_top2", "uniform_all3"):
                    def replace_gate(module, inputs, outputs):
                        sparse, dense = outputs
                        weights = (sparse > 0).to(sparse.dtype) / 2 if variant == "equal_selected_top2" else torch.full_like(sparse, 1 / 3)
                        return weights, dense
                    hook = model.gate.register_forward_hook(replace_gate)
                    try:
                        pred = tiled_forward(model, noisy, sigma=level).clamp(0, 1)
                        interventions.write(image=path.name, variant=variant, psnr_out=psnr(pred, target))
                        del pred
                    finally:
                        hook.remove()
                h, w = hr.shape[-2:]
                y, x = (h - 256) // 2, (w - 256) // 2
                for sample in range(args.routing_samples):
                    seed = idx if sample == 0 else 10000 + idx * args.routing_samples + sample
                    degraded, sampled = (lr, params) if sample == 0 else degrade(hr[0], np.random.default_rng(seed))
                    crop = degraded[None, :, y:y + 256, x:x + 256].cuda()
                    _, aux = model(crop, return_aux=True)
                    stats = routing_stats(aux["routing"], aux["dense_routing"])
                    routing.append({"image": path.name, "seed": seed, **sampled, **stats})
                    if sample == 0:
                        plot_routing(crop, aux["routing"], out / f"routing_{path.stem}.png")
                    del aux, crop
                (out / "routing.json").write_text(json.dumps(routing, indent=2, allow_nan=False) + "\n")
            del noisy, target, hr, lr
            print(f"{method}: scored {path.name}, tiles 128/256/512", flush=True)
        result["tile_psnr"] = {str(size): float(np.mean(values)) for size, values in scores.items()}
        summary["models"][method] = result
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        del model
        torch.cuda.empty_cache()
    summary["routing_correlations"] = correlations(routing)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
