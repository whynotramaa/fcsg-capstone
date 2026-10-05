"""One config-driven training entry point for every model in the project.

Phase 4 ablations become config diffs rather than forked scripts, which is also
how you keep them honest.

    python training/train.py --config configs/dncnn.toml --data /kaggle/input \
        --out /kaggle/working/ckpt
"""

import argparse
import math
import random
import sys
import time
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fcsg_net.data import PatchDataset, TileDataset  # noqa: E402
from fcsg_net.degrade import DegradedDataset  # noqa: E402
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
    if name == "ffdnet":
        from models.ffdnet import FFDNet

        return FFDNet(**kwargs)
    if name == "fcsg":
        from models.fcsg import FCSGNet

        return FCSGNet(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def predict(model, noisy, sigma):
    from models.ffdnet import FFDNet

    return model(noisy, sigma) if isinstance(model, FFDNet) else model(noisy)


def fixed_sigma(cfg):
    value = cfg.get("ffdnet_sigma", "oracle")
    return None if value == "oracle" else value / 255


def charbonnier(pred, target, eps=1e-3):
    """Standard restoration loss. Used for the baselines too, so the Phase 3
    comparison against FCSG-Net is like for like."""
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def freq_loss(pred, target):
    return torch.fft.rfft2((pred - target).float(), norm="ortho").abs().mean()


def routing_entropy_loss(dense):
    p = dense.float().clamp_min(1e-8)
    return (p * p.log()).sum(2).mean((0, 2, 3)).sum()


def cosine_lr(step, total, lr, lr_min):
    """Computed from the step, so resuming needs no scheduler state."""
    t = min(step / max(total, 1), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def validate(model, val_batch, sigma, device, clip=True, blind=None):
    """Fixed crops, fixed noise, so numbers are comparable across sessions."""
    model.eval()
    hr = val_batch["hr"].to(device)
    g = torch.Generator(device=device).manual_seed(VAL_SEED)
    noisy = val_batch["lr"].to(device) if "lr" in val_batch else add_noise(hr, sigma, generator=g, clip=clip)
    levels = val_batch["noise_sigma"].to(device) / 255 if "lr" in val_batch else sigma
    levels = levels if blind is None else blind
    out = predict(model, noisy, levels)
    model.train()
    return psnr(noisy, hr, clip=False), psnr(out, hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="any ancestor of the DIV2K image directories")
    ap.add_argument("--train-dir", help="skip the search and use this directory for training")
    ap.add_argument("--val-dir", help="skip the search and use this directory for validation")
    ap.add_argument("--tiles", help="a .npy tile cache from build_tiles.py; much faster than --data")
    ap.add_argument("--out", required=True, help="checkpoints and log land here")
    ap.add_argument("--steps", type=int, help="override config, for smoke runs")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--overfit-one-image", action="store_true")
    ap.add_argument("--max-hours", type=float, help="override config, stop and save before this")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    for k in ("steps", "batch", "max_hours"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if "seed" in cfg:
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        print(f"training seed={cfg['seed']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} config={args.config} steps={cfg['steps']} batch={cfg['batch']}")
    if device == "cpu":
        print("WARNING: no GPU visible. Enable the accelerator before a real run.")

    train_dir, val_dir = resolve_div2k(args.data, args.train_dir, args.val_dir)
    print(f"train={train_dir}\nval={val_dir}")

    sigma = cfg["sigma"] / 255.0
    if cfg.get("degradation", "gaussian") not in ("gaussian", "composite"):
        raise ValueError("degradation must be 'gaussian' or 'composite'")
    composite = cfg.get("degradation", "gaussian") == "composite"
    clip = cfg.get("clip_noise", True)
    blind = fixed_sigma(cfg)
    if args.overfit_one_image:
        # One fixed crop with one fixed noise realisation, repeated forever. A
        # model that cannot memorise this has a bug, and finding out costs two
        # minutes here instead of ten hours on a real run.
        one = PatchDataset(train_dir, patch=cfg["patch"], seed=0, limit=1)
        one = DegradedDataset(one, seed=0) if composite else one
        train_batch = next(iter(DataLoader(one, batch_size=1)))
        loader = None
    else:
        workers = cfg.get("workers", 4)
        tiles = args.tiles or cfg.get("tiles")
        train_ds = TileDataset(tiles) if tiles else PatchDataset(train_dir, patch=cfg["patch"])
        if composite:
            train_ds = DegradedDataset(train_ds)
        if len(train_ds) < cfg["batch"]:
            raise ValueError("training dataset is smaller than the batch size")
        print(f"train samples={len(train_ds)} source={tiles or train_dir}")
        loader = DataLoader(
            train_ds,
            batch_size=cfg["batch"],
            shuffle=True,
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=workers > 0,
        )

    val_ds = PatchDataset(val_dir, patch=cfg["patch"], seed=VAL_SEED, limit=cfg.get("val_images", 16))
    if composite:
        val_ds = DegradedDataset(val_ds, seed=VAL_SEED)
    val_batch = next(iter(DataLoader(val_ds, batch_size=len(val_ds))))

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

    log = CSVLog(out_dir / "train_log.csv", ["step", "loss", "lr", "psnr_in", "psnr_out", "secs", "entropy"])
    freq_weight, ent_weight = cfg.get("freq_weight", 0.0), cfg.get("ent_weight", 0.0)
    if ent_weight:
        uniform = torch.full((1, 3, 3, 1, 1), 1 / 3)
        peaked = torch.tensor([0.98, 0.01, 0.01]).reshape(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1)
        assert routing_entropy_loss(uniform) < routing_entropy_loss(peaked), "entropy term has the wrong sign"
    total = cfg["steps"]
    # A Kaggle session killed at the 12h cap loses /kaggle/working entirely, so
    # the run has to stop itself with time to spare and let the notebook finish.
    budget = cfg.get("max_hours", 11.5) * 3600
    if start >= total:
        print(f"already at step {start}/{total}, nothing to do")
        return
    data = infinite(loader) if loader is not None else None
    fixed_noisy = None
    t0 = time.time()

    for step in range(start + 1, total + 1):
        lr = cosine_lr(step, total, cfg["lr"], cfg["lr_min"])
        for g in opt.param_groups:
            g["lr"] = lr

        if loader is None:
            hr = train_batch["hr"].to(device)
            if fixed_noisy is None:
                g = torch.Generator(device=device).manual_seed(0)
                fixed_noisy = train_batch["lr"].to(device) if composite else add_noise(hr, sigma, generator=g, clip=clip)
            noisy = fixed_noisy
            levels = train_batch["noise_sigma"].to(device) / 255 if composite else sigma
            levels = levels if blind is None else blind
        else:
            sample = next(data)
            hr = sample["hr"].to(device, non_blocking=True)
            noisy = sample["lr"].to(device, non_blocking=True) if composite else add_noise(hr, sigma, clip=clip)
            levels = sample["noise_sigma"].to(device) / 255 if composite else sigma
            levels = levels if blind is None else blind

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            entropy = None
            if ent_weight:
                pred, aux = model(noisy, return_aux=True)
                entropy = routing_entropy_loss(aux["dense_routing"])
            else:
                pred = predict(model, noisy, levels)
            loss = charbonnier(pred, hr)
            if freq_weight:
                loss = loss + freq_weight * freq_loss(pred, hr)
            if entropy is not None:
                loss = loss + ent_weight * max(0.0, 1 - step / total) * entropy

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 50) == 0 or step == total:
            print(f"step {step}/{total}  loss {loss.item():.5f}  lr {lr:.2e}  {time.time() - t0:.0f}s")
            log.write(step=step, loss=f"{loss.item():.6f}", lr=f"{lr:.3e}", secs=f"{time.time() - t0:.0f}",
                      entropy="" if entropy is None else f"{-entropy.item():.4f}")

        if step % cfg.get("val_every", 2000) == 0 or step == total:
            p_in, p_out = validate(model, val_batch, sigma, device, clip=clip, blind=blind)
            print(f"  val psnr  in {p_in:.2f} dB  ->  out {p_out:.2f} dB")
            log.write(step=step, psnr_in=f"{p_in:.3f}", psnr_out=f"{p_out:.3f}", secs=f"{time.time() - t0:.0f}")

        out_of_time = time.time() - t0 > budget
        if step % cfg.get("ckpt_every", 2000) == 0 or step == total or out_of_time:
            save_checkpoint(out_dir / f"ckpt_{step:07d}.pt", step, model, opt, scaler,
                            extra={"config": cfg, "params": n_params})
            print(f"  saved ckpt_{step:07d}.pt")

        if out_of_time:
            rate = (step - start) / max(time.time() - t0, 1)
            print(f"stopping at step {step}/{total}: {budget / 3600:.1f}h budget reached. "
                  f"Re-run the notebook with this output attached to resume. "
                  f"~{(total - step) / max(rate, 1e-9) / 3600:.1f}h of GPU left at this rate.")
            break

    print(f"stopped at step {step}/{total} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
