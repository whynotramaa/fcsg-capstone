"""Evaluate a checkpoint on full DIV2K validation images and record the result.

Appends one row to results/benchmark.csv and writes the qualitative figure. The
same CSV is what the Phase 3 comparison table fills, so FFDNet and FCSG-Net just
append rows to it later.

    python evaluation/eval.py --ckpt /kaggle/working/ckpt/ckpt_0000200.pt \
        --data /kaggle/input --out results --notes smoke
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fcsg_net.data import find_images  # noqa: E402
from fcsg_net.metrics import psnr  # noqa: E402
from fcsg_net.utils import CSVLog, add_noise, resolve_div2k  # noqa: E402

sys.path.append(str(ROOT / "training"))
from train import build_model, predict  # noqa: E402


@torch.no_grad()
def tiled_forward(model, x, tile=256, overlap=32, sigma=25 / 255):
    """Full-image inference in overlapping tiles.

    FFT bands and spatial gates depend on context. Tiling is an approximation,
    even with cutoffs in cycles/pixel. Record the tile policy with each result.
    """
    if not 0 <= overlap < tile:
        raise ValueError("require 0 <= overlap < tile")
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        return predict(model, x, sigma)
    stride = tile - overlap
    out = torch.zeros_like(x)
    count = torch.zeros_like(x)
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for xx in xs:
            patch = x[:, :, y : y + tile, xx : xx + tile]
            levels = sigma
            if isinstance(sigma, torch.Tensor) and sigma.ndim == 4 and sigma.shape[-2:] == (h, w):
                levels = sigma[..., y : y + tile, xx : xx + tile]
            out[:, :, y : y + tile, xx : xx + tile] += predict(model, patch, levels)
            count[:, :, y : y + tile, xx : xx + tile] += 1
    return out / count.clamp_min(1)


def load_image(path, device, multiple=1):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)
    # Models handle odd sizes internally. Keep all pixels for published benchmarks.
    h, w = t.shape[1] // multiple * multiple, t.shape[2] // multiple * multiple
    return t[:, :h, :w].unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt")
    source.add_argument("--ffdnet-weights", help="author's KAIR ffdnet_color.pth")
    ap.add_argument("--author-protocol", action="store_true", help="FFDNet CBSD68: unclipped AWGN, seed 0 per image, uint8 output, full image")
    ap.add_argument("--assert-psnr", type=float, help="require agreement within 0.5 dB")
    ap.add_argument("--data", required=True, help="any ancestor of the DIV2K image directories")
    ap.add_argument("--val-dir", help="skip the search and score this directory")
    ap.add_argument("--out", default="results", help="benchmark.csv and figures/ go here")
    ap.add_argument("--limit", type=int, default=20, help="validation images to score")
    ap.add_argument("--notes", default="", help="e.g. smoke, final")
    ap.add_argument("--method", default="", help="defaults to the model name in the checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ffdnet_weights:
        ck = {"step": 0, "model": torch.load(args.ffdnet_weights, map_location=device, weights_only=True),
              "extra": {"config": {"model": "ffdnet", "sigma": 25, "clip_noise": False}}}
    else:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["extra"]["config"]
    method = args.method or cfg["model"]
    sigma = cfg["sigma"] / 255.0

    model = build_model(cfg["model"], **cfg.get("model_args", {})).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{method} @ step {ck['step']}  params {n_params:,}  sigma {cfg['sigma']}  device {device}")

    val_dir = Path(args.val_dir) if args.val_dir else resolve_div2k(args.data)[1]
    files = find_images(val_dir)[: args.limit]
    if args.author_protocol and (cfg["model"] != "ffdnet" or len(files) != 68):
        raise ValueError("author protocol requires FFDNet and all 68 clean CBSD68 images")
    if not files:
        raise ValueError("no evaluation images selected")

    ins, outs, samples = [], [], []
    for i, f in enumerate(files):
        hr = load_image(f, device)
        g = torch.Generator(device=device).manual_seed(i)
        if args.author_protocol:
            import numpy as np

            noise = np.random.RandomState(0).normal(0, sigma, (*hr.shape[-2:], 3)).astype(np.float32)
            noisy = hr + torch.from_numpy(noise).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = predict(model, noisy, sigma).clamp(0, 1).mul(255).round().div(255)
        elif cfg.get("degradation") == "composite":
            import numpy as np
            from fcsg_net.degrade import degrade

            lr, params = degrade(hr[0].cpu(), np.random.default_rng(i))
            noisy = lr.unsqueeze(0).to(device)
            pred = tiled_forward(model, noisy, sigma=params["noise_sigma"] / 255).clamp(0, 1)
        else:
            noisy = add_noise(hr, sigma, generator=g, clip=cfg.get("clip_noise", True))
            pred = tiled_forward(model, noisy, sigma=sigma).clamp(0, 1)
        p_in, p_out = psnr(noisy, hr, clip=False), psnr(pred, hr)
        ins.append(p_in)
        outs.append(p_out)
        print(f"  {f.name}: {p_in:.2f} -> {p_out:.2f} dB")
        if len(samples) < 2:
            samples.append((noisy, pred, hr, p_in, p_out))

    psnr_in, psnr_out = sum(ins) / len(ins), sum(outs) / len(outs)
    print(f"\nmean over {len(files)} images: {psnr_in:.2f} dB in -> {psnr_out:.2f} dB out "
          f"(+{psnr_out - psnr_in:.2f})")

    out_dir = Path(args.out)
    log = CSVLog(out_dir / "benchmark.csv",
                 ["method", "steps", "sigma", "psnr_in", "psnr_out", "gain", "params", "images", "notes"])
    log.write(method=method, steps=ck["step"], sigma=cfg["sigma"],
              psnr_in=f"{psnr_in:.3f}", psnr_out=f"{psnr_out:.3f}",
              gain=f"{psnr_out - psnr_in:.3f}", params=n_params,
              images=len(files), notes=args.notes)
    print(f"appended row to {out_dir / 'benchmark.csv'}")

    save_qualitative(samples, out_dir / "figures" / "qualitative.png")
    result = {"method": method, "steps": ck["step"], "images": len(files),
              "psnr": psnr_out, "params": n_params, "notes": args.notes,
              "protocol": "author-cbsd68" if args.author_protocol else "tiled-256-overlap32",
              "source": args.ffdnet_weights or args.ckpt}
    if args.assert_psnr is not None:
        result.update(target=args.assert_psnr, gap=abs(psnr_out - args.assert_psnr),
                      passed=abs(psnr_out - args.assert_psnr) <= 0.5)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.assert_psnr is not None:
        assert result["passed"], result


def save_qualitative(samples, path, crop=320):
    """Noisy / restored / clean strip. The figure a panel actually looks at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3.1 * len(samples)), squeeze=False)
    for r, (noisy, pred, hr, p_in, p_out) in enumerate(samples):
        panels = [(noisy, f"noisy  {p_in:.2f} dB"), (pred, f"restored  {p_out:.2f} dB"), (hr, "ground truth")]
        for c, (img, title) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img[0, :, :crop, :crop].permute(1, 2, 0).cpu().numpy())
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
