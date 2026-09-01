"""DnCNN (Zhang et al., TIP 2017)."""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels=3, depth=17, width=64):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.body(x)


def build(**kwargs):
    return DnCNN(**kwargs)


if __name__ == "__main__":
    m = DnCNN()
    n = sum(p.numel() for p in m.parameters())
    print(f"DnCNN params: {n:,}")
    # 17 layers at width 64 is ~0.56M. A silent width or depth typo shows up here
    # rather than after a ten-hour run.
    assert 550_000 < n < 570_000, n
    x = torch.rand(2, 3, 64, 64)
    y = m(x)
    assert y.shape == x.shape, y.shape
    assert torch.isfinite(y).all()
    print("dncnn ok")
