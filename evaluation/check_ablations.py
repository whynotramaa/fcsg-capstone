"""Small GPU check before the matched reference and uniform-routing runs."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from models.fcsg import FCSGNet


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run ablation checks on Kaggle/Colab")
    torch.manual_seed(1234)
    reference = FCSGNet().cuda()
    torch.manual_seed(1234)
    uniform = FCSGNet(uniform_routing=True).cuda()
    # Keep the unused gate module so initialization stays identical in both runs.
    assert all(torch.equal(value, uniform.state_dict()[name]) for name, value in reference.state_dict().items())
    x = torch.rand(1, 3, 128, 128, device="cuda")
    _, aux = reference(x, return_aux=True)
    assert ((aux["routing"] > 0).sum(2) == 2).all()
    pred, aux = uniform(x, return_aux=True)
    assert pred.shape == x.shape and torch.isfinite(pred).all()
    assert torch.allclose(aux["routing"], torch.full_like(aux["routing"], 1 / 3))
    assert torch.allclose(aux["routing"].sum(2), torch.ones_like(aux["routing"][:, :, 0]))
    pred.square().mean().backward()
    assert all(expert.head.weight.grad.abs().sum() > 0 for expert in uniform.experts)
    assert all(p.grad is None for p in uniform.gate.parameters())
    uniform.uniform_routing = False
    with torch.no_grad():
        assert torch.equal(reference(x), uniform(x)), "default learned routing changed"
    Path("/kaggle/working/ablation_checks.json").write_text(json.dumps({
        "passed": True, "seed": 1234, "identical_initialization": True,
        "uniform_all3": True, "expert_gradients": True, "default_routing_preserved": True,
    }, indent=2) + "\n")
    print("Matched initialization, uniform routing, expert gradients, and default routing checks passed.", flush=True)


if __name__ == "__main__":
    main()
