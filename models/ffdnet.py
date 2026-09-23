"""Colour FFDNet, compatible with the author's KAIR ffdnet_color.pth weights.

Reference: https://github.com/cszn/KAIR/blob/master/models/network_ffdnet.py
The author released inference weights with batch normalization folded into convs.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FFDNet(nn.Module):
    def __init__(self, channels=3, depth=12, width=96):
        super().__init__()
        if channels != 3 or depth < 2 or width < 1:
            raise ValueError("FFDNet expects RGB, depth >= 2, and positive width")
        layers = [nn.Conv2d(channels * 4 + 1, width, 3, padding=1), nn.ReLU(True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(True)]
        layers.append(nn.Conv2d(width, channels * 4, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x, sigma):
        """sigma is in [0, 1] units: scalar, (B,), or (B, 1, H, W)."""
        h, w = x.shape[-2:]
        z = F.pixel_unshuffle(F.pad(x, (0, w % 2, 0, h % 2), mode="replicate"), 2)
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.ndim <= 1:
            sigma = sigma.reshape(-1, 1, 1, 1)
        if sigma.ndim != 4 or sigma.shape[1] != 1 or sigma.shape[0] not in (1, x.shape[0]):
            raise ValueError("sigma must be scalar, per-image, or a single-channel noise map")
        if not torch.isfinite(sigma).all() or (sigma < 0).any() or (sigma > 1).any():
            raise ValueError("sigma must be finite and between 0 and 1")
        sigma = F.interpolate(sigma, size=z.shape[-2:], mode="bilinear", align_corners=False)
        z = self.model(torch.cat((z, sigma.expand(x.shape[0], -1, -1, -1)), dim=1))
        return F.pixel_shuffle(z, 2)[..., :h, :w]
