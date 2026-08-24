"""One config-driven training entry point for every model in the project.

Phase 4 ablations become config diffs rather than forked scripts, which is also
how you keep them honest.

    python training/train.py --config configs/dncnn.toml --data /kaggle/input \
        --out /kaggle/working/ckpt
"""

import argparse
import math
import sys
import time
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fcsg_net.data import PatchDataset  # noqa: E402
from fcsg_net.metrics import psnr  # noqa: E402
from fcsg_net.utils import (  # noqa: E402
    CSVLog,
    add_noise,
    load_checkpoint,
    newest_checkpoint,
    resolve_div2k,
    save_checkpoint,
)

VAL_SEED = 1234


def build_model(name, **kwargs):
    if name == "dncnn":
        from models.dncnn import DnCNN

        return DnCNN(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def charbonnier(pred, target, eps=1e-3):
    """Standard restoration loss. Used for the baselines too, so the Phase 3
    comparison against FCSG-Net is like for like."""
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def cosine_lr(step, total, lr, lr_min):
    """Computed from the step, so resuming needs no scheduler state."""
    t = min(step / max(total, 1), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def validate(model, val_batch, sigma, device):
    """Fixed crops, fixed noise, so numbers are comparable across sessions."""
    model.eval()
    hr = val_batch.to(device)
    g = torch.Generator(device=device).manual_seed(VAL_SEED)
    noisy = add_noise(hr, sigma, generator=g)
    out = model(noisy)
    model.train()
    return psnr(noisy, hr), psnr(out, hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="any ancestor of DIV2K_train_HR")
    ap.add_argument("--out", required=True, help="checkpoints and log land here")
    ap.add_argument("--steps", type=int, help="override config, for smoke runs")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--overfit-one-image", action="store_true")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    for k in ("steps", "batch"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} config={args.config} steps={cfg['steps']} batch={cfg['batch']}")
    if device == "cpu":
        print("WARNING: no GPU visible. Enable the accelerator before a real run.")

    train_dir, val_dir = resolve_div2k(args.data)
    print(f"train={train_dir}\nval={val_dir}")

    sigma = cfg["sigma"] / 255.0
    if args.overfit_one_image:
        # One fixed crop with one fixed noise realisation, repeated forever. A
        # model that cannot memorise this has a bug, and finding out costs two
        # minutes here instead of ten hours on a real run.
        one = PatchDataset(train_dir, patch=cfg["patch"], seed=0, limit=1)[0]["hr"]
        train_batch = one.unsqueeze(0).repeat(cfg["batch"], 1, 1, 1)
        loader = None
    else:
        workers = cfg.get("workers", 4)
        loader = DataLoader(
            PatchDataset(train_dir, patch=cfg["patch"]),
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=workers > 0,
        )

    val_ds = PatchDataset(val_dir, patch=cfg["patch"], seed=VAL_SEED, limit=cfg.get("val_images", 16))
    val_batch = torch.stack([val_ds[i]["hr"] for i in range(len(val_ds))])

    model = build_model(cfg["model"], **cfg.get("model_args", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model']} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    # float16, not bfloat16: neither the P100 nor the T4 supports bf16.
    use_amp = device == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    latest = newest_checkpoint(out_dir)
    if latest:
        start = load_checkpoint(latest, model, opt, scaler, device=device)
        print(f"RESUMED from {latest.name} at step {start}")
    else:
        print("no checkpoint found, starting from step 0")

    log = CSVLog(out_dir / "train_log.csv", ["step", "loss", "lr", "psnr_in", "psnr_out", "secs"])
    total = cfg["steps"]
    data = infinite(loader) if loader is not None else None
    fixed_noisy = None
    t0 = time.time()

    for step in range(start + 1, total + 1):
        lr = cosine_lr(step, total, cfg["lr"], cfg["lr_min"])
        for g in opt.param_groups:
            g["lr"] = lr

        if loader is None:
            hr = train_batch.to(device)
            if fixed_noisy is None:
                g = torch.Generator(device=device).manual_seed(0)
                fixed_noisy = add_noise(hr, sigma, generator=g)
            noisy = fixed_noisy
        else:
            hr = next(data)["hr"].to(device, non_blocking=True)
            noisy = add_noise(hr, sigma)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = charbonnier(model(noisy), hr)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 50) == 0 or step == total:
            print(f"step {step}/{total}  loss {loss.item():.5f}  lr {lr:.2e}  {time.time() - t0:.0f}s")
            log.write(step=step, loss=f"{loss.item():.6f}", lr=f"{lr:.3e}", secs=f"{time.time() - t0:.0f}")

        if step % cfg.get("val_every", 2000) == 0 or step == total:
            p_in, p_out = validate(model, val_batch, sigma, device)
            print(f"  val psnr  in {p_in:.2f} dB  ->  out {p_out:.2f} dB")
            log.write(step=step, psnr_in=f"{p_in:.3f}", psnr_out=f"{p_out:.3f}", secs=f"{time.time() - t0:.0f}")

        if step % cfg.get("ckpt_every", 2000) == 0 or step == total:
            save_checkpoint(out_dir / f"ckpt_{step:07d}.pt", step, model, opt, scaler,
                            extra={"config": cfg, "params": n_params})
            print(f"  saved ckpt_{step:07d}.pt")

    print(f"done at step {total} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
