"""Frequency selection followed by shared spatial experts and sparse mixing."""

import torch
from torch import nn

from models.blocks import Expert, FrequencyDecompose, Gate, SEFusion


class FCSGNet(nn.Module):
    def __init__(self, width=48, blocks=4, tau=1.0, top_k=2, uniform_routing=False):
        super().__init__()
        if width < 1 or blocks < 1:
            raise ValueError("width and blocks must be positive")
        self.decompose = FrequencyDecompose()
        self.experts = nn.ModuleList(Expert(k, width, blocks) for k in (7, 5, 3))
        self.gate = Gate(tau=tau, top_k=top_k)
        self.uniform_routing = uniform_routing
        # ponytail: retain 681 unused gate parameters to match initialization; remove for a standalone uniform model.
        if uniform_routing:
            self.gate.requires_grad_(False)
        self.fusion = SEFusion()
        self.refine = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 3, 3, padding=1))

    def forward(self, x, return_aux=False):
        bands = self.decompose(x)
        if self.uniform_routing:
            routing = bands.new_full((bands.shape[0], 3, 3, *bands.shape[-2:]), 1 / 3)
            dense = routing
        else:
            routing, dense = self.gate(bands)
        b, _, c, h, w = bands.shape
        flat = bands.reshape(b * 3, c, h, w)
        mixed = torch.zeros_like(bands)
        # ponytail: all experts execute; sparse dispatch needs tiled expert execution.
        for e, expert in enumerate(self.experts):
            mixed = mixed + expert(flat).reshape(b, 3, c, h, w) * routing[:, :, e, None]
        out = x + self.refine(self.fusion(mixed.reshape(b, 9, h, w)))
        if return_aux:
            return out, {"bands": bands, "routing": routing, "dense_routing": dense,
                         "band_outputs": mixed}
        return out
