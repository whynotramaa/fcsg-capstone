"""Metrics."""

import torch


def psnr(pred, target, max_val=1.0, clip=True):
    """Mean RGB PSNR. Set clip=False to measure an unclipped noisy input."""
    pred = (pred.clamp(0, max_val) if clip else pred).float()
    target = target.clamp(0, max_val).float()
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(max_val**2 / mse.clamp_min(1e-12))).mean().item()
