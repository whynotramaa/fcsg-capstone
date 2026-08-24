"""Metrics. PSNR only until the Phase 3 benchmark table needs SSIM and LPIPS."""

import torch


def psnr(pred, target, max_val=1.0):
    """Mean PSNR in dB over the batch. Inputs are (B, C, H, W) in [0, 1]."""
    pred = pred.clamp(0, max_val).float()
    target = target.clamp(0, max_val).float()
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(max_val**2 / mse.clamp_min(1e-12))).mean().item()
