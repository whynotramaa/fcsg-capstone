"""Frequency bands, shared experts, spatial routing, and SE fusion."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyDecompose(nn.Module):
    def __init__(self, low=0.10, high=0.25, transition=0.05):
        super().__init__()
        if not (0 < transition < min(2 * low, high - low) and low < high < 0.5):
            raise ValueError("require 0 < low < high < 0.5 and nonoverlapping transitions")
        self.low, self.high, self.transition = low, high, transition

    def masks(self, h, w, device):
        # fftfreq is already in the unshifted ordering used by rfft2.
        fy = torch.fft.fftfreq(h, device=device)[:, None]
        fx = torch.fft.rfftfreq(w, device=device)[None, :]
        radius = torch.sqrt(fy.square() + fx.square())
        def lowpass(cutoff):
            t = ((radius - cutoff) / self.transition + 0.5).clamp(0, 1)
            return 0.5 * (1 + torch.cos(math.pi * t))
        low, upper = lowpass(self.low), lowpass(self.high)
        return torch.stack((low, upper - low, 1 - upper))

    def forward(self, x):
        # CUDA half FFTs require powers of two. Float32 also preserves the sum.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            masks = self.masks(*x.shape[-2:], x.device)
            bands = torch.fft.irfft2(spectrum[:, None] * masks[None, :, None],
                                    s=x.shape[-2:], norm="ortho")
        return bands


class ResidualBlock(nn.Module):
    def __init__(self, width, kernel):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width),
            nn.Conv2d(width, width, 1), nn.ReLU(),
            nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class Expert(nn.Module):
    def __init__(self, kernel, width=48, blocks=4):
        super().__init__()
        # Lossless rearrangement and separable convs target the 10 GFLOP budget.
        self.head = nn.Conv2d(12, width, 1)
        self.body = nn.Sequential(*(ResidualBlock(width, kernel) for _ in range(blocks)))
        self.tail = nn.Conv2d(width, 12, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        z = F.pixel_unshuffle(F.pad(x, (0, w % 2, 0, h % 2), mode="replicate"), 2)
        z = z + self.tail(self.body(F.relu(self.head(z))))
        return F.pixel_shuffle(z, 2)[..., :h, :w]


class Gate(nn.Module):
    def __init__(self, window=16, tau=1.0, top_k=2):
        super().__init__()
        if window < 1 or tau <= 0 or not 1 <= top_k <= 3:
            raise ValueError("invalid gate window, temperature, or top_k")
        self.window, self.tau, self.top_k = window, tau, top_k
        self.net = nn.Sequential(nn.Conv2d(18, 24, 1), nn.ReLU(), nn.Conv2d(24, 9, 1))

    def forward(self, bands):
        b, _, _, h, w = bands.shape
        z = bands.reshape(b, 9, h, w)
        window = min(self.window, h, w)
        mean = F.avg_pool2d(z, window, window, ceil_mode=True)
        energy = F.avg_pool2d(z.square(), window, window, ceil_mode=True)
        logits = self.net(torch.cat((mean, energy), 1)) / self.tau
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits.reshape(b, 3, 3, h, w)
        dense = logits.softmax(dim=2)
        values, indices = logits.topk(self.top_k, dim=2)
        sparse = torch.zeros_like(logits).scatter(2, indices, values.softmax(dim=2))
        return sparse, dense


class SEFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(9, 3, 1),
                                       nn.ReLU(), nn.Conv2d(3, 9, 1), nn.Sigmoid())
        self.project = nn.Conv2d(9, 3, 1)

    def forward(self, x):
        return self.project(x * self.attention(x))
