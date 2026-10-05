"""Metrics."""

import copy
import math

import torch
from torch.nn import functional as F


def psnr(pred, target, max_val=1.0, clip=True):
    """Mean RGB PSNR. Set clip=False to measure an unclipped noisy input."""
    pred = (pred.clamp(0, max_val) if clip else pred).float()
    target = target.clamp(0, max_val).float()
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(max_val**2 / mse.clamp_min(1e-12))).mean().item()


def ssim(pred, target, max_val=1.0):
    pred, target = pred.clamp(0, max_val).float(), target.clamp(0, max_val).float()
    g = torch.exp(-((torch.arange(11, device=pred.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).expand(pred.shape[1], 1, 11, 11).contiguous()
    blur = lambda x: F.conv2d(x, window, groups=x.shape[1])
    mu_x, mu_y = blur(pred), blur(target)
    var_x = blur(pred * pred) - mu_x**2
    var_y = blur(target * target) - mu_y**2
    cov = blur(pred * target) - mu_x * mu_y
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return s.mean().item()


def gflops(model, name, size=256):
    from thop import profile

    device = next(model.parameters()).device
    x = torch.rand(1, 3, size, size, device=device)
    inputs = (x, torch.full((1,), 25 / 255, device=device)) if name == "ffdnet" else (x,)
    with torch.no_grad():
        macs, _ = profile(copy.deepcopy(model), inputs=inputs, verbose=False)
    fft = 12 * 5 * size * size * math.log2(size * size) if name == "fcsg" else 0
    return (2 * macs + fft) / 1e9
