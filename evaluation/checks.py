"""Executable Phase 1/2 checks. Run on Kaggle/Colab, before long training."""

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fcsg_net.data import PatchDataset, TileDataset
from fcsg_net.degrade import DegradedDataset, degrade, snapshot
from fcsg_net.metrics import psnr
from fcsg_net.utils import add_noise, load_checkpoint, save_checkpoint
from models.blocks import FrequencyDecompose
from models.fcsg import FCSGNet
from models.ffdnet import FFDNet
from training.train import build_model, charbonnier, predict, validate
from evaluation.eval import tiled_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training HR directory")
    ap.add_argument("--out", default="results/phase2_checks.json")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU checks require an active CUDA runtime")
        memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 4_000_000_000 / memory))
    result = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "torch": torch.__version__, "seed": 7}
    assert abs(psnr(torch.full((1, 3, 2, 2), 2.0), torch.zeros(1, 3, 2, 2), clip=False)
               + 20 * math.log10(2)) < 1e-5

    split = FrequencyDecompose().to(device)
    errors = []
    for shape in ((1, 3, 32, 48), (1, 3, 65, 67), (1, 3, 128, 128), (1, 3, 256, 256)):
        x = torch.rand(shape, device=device)
        bands = split(x)
        errors.append((bands.sum(1) - x).abs().max().item())
        assert torch.allclose(bands.sum(1), x, atol=1e-5), shape
        assert split.masks(*shape[-2:], device).min() >= 0
    # Same cycles/pixel, different FFT bin index. Low/mid/high pure tones.
    for band, freq in enumerate((1 / 16, 3 / 16, 3 / 8)):
        energies = []
        for size in (128, 256):
            tone = torch.cos(2 * math.pi * freq * torch.arange(size, device=device))
            x = tone[None, None, None].expand(1, 3, size, size)
            energies.append(split(x).square().mean((0, 2, 3, 4)))
        assert torch.allclose(energies[0], energies[1], atol=1e-5)
        assert energies[0].argmax().item() == band
    result["reconstruction_max_error"] = max(errors)
    result["frequency_units_check"] = True

    ffd = FFDNet().to(device)
    odd = torch.rand(2, 3, 31, 35, device=device)
    low = ffd(odd, torch.tensor([5 / 255, 50 / 255], device=device))
    high = ffd(odd, torch.full((2, 1, 31, 35), 0.3, device=device))
    assert low.shape == odd.shape and torch.isfinite(low).all()
    assert not torch.allclose(low, high), "FFDNet ignores the noise map"
    low.square().mean().backward()
    assert ffd.model[0].weight.grad[:, -1].abs().sum() > 0
    assert sum(p.numel() for p in ffd.parameters()) == 852108
    try:
        ffd(odd, -1)
        raise AssertionError("negative noise level accepted")
    except ValueError:
        pass
    result["ffdnet_noise_map_check"] = True
    del ffd, low, high

    clean = PatchDataset(args.data, patch=128, seed=7, limit=4)
    hr = clean[0]["hr"]
    a, pa = degrade(hr, np.random.default_rng(7))
    b, pb = degrade(hr, np.random.default_rng(7))
    assert torch.equal(a, b) and pa == pb
    assert a.shape == hr.shape and 0 <= a.min() <= a.max() <= 1
    assert 0.5 <= pa["blur_sigma"] <= 2 and 5 <= pa["noise_sigma"] <= 50
    assert 30 <= pa["jpeg_quality"] <= 95 and 1 <= pa["scale"] <= 2
    assert not torch.equal(a, degrade(hr, np.random.default_rng(8))[0])
    assert degrade(hr, np.random.default_rng(7), resample=False)[1]["scale"] == 1
    degraded = DegradedDataset(clean, seed=7)
    batch = next(iter(DataLoader(degraded, batch_size=2, num_workers=2)))
    assert batch["lr"].shape == batch["hr"].shape and batch["noise_sigma"].shape == (2,)
    assert torch.equal(batch["lr"][0], a)
    with tempfile.TemporaryDirectory() as tmp:
        tiles = Path(tmp) / "tiles.npy"
        np.save(tiles, np.stack([np.rint(hr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)] * 4))
        ds = DegradedDataset(TileDataset(tiles, seed=7), seed=7)
        assert ds[0]["lr"].shape == hr.shape
        t0 = time.perf_counter()
        for i in range(32):
            ds[i % len(ds)]
        result["degradation_samples_per_second"] = 32 / (time.perf_counter() - t0)
        from PIL import Image

        source = Path(tmp) / "source"
        source.mkdir()
        Image.fromarray(np.rint(hr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(source / "one.png")
        first, second = Path(tmp) / "snapshot1", Path(tmp) / "snapshot2"
        snapshot(source, first, count=2, seed=7)
        snapshot(source, second, count=2, seed=7)
        for path in first.rglob("*"):
            if path.is_file():
                assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    result["degradation_reproducible"] = True

    model = FCSGNet().to(device)
    x = torch.rand(1, 3, 256, 256, device=device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    out, aux = model(x, return_aux=True)
    assert out.shape == x.shape and torch.isfinite(out).all()
    weights = aux["routing"]
    assert weights.shape == (1, 3, 3, 256, 256)
    assert torch.allclose(weights.sum(2), torch.ones_like(weights[:, :, 0]), atol=1e-6)
    assert ((weights > 0).sum(2) == 2).all()
    assert weights.std(dim=(-1, -2)).max() > 0, "routing is spatially constant"
    out.square().mean().backward()
    assert model.gate.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if device == "cuda" else None
    assert result["parameters"] < 5_000_000
    if device == "cuda":
        assert result["peak_reserved_bytes"] < 4_000_000_000
    del out, aux, weights
    model.zero_grad(set_to_none=True)
    from thop import profile

    macs, _ = profile(model, inputs=(x,), verbose=False)
    # thop counts conv MACs, not FFTs. Conservatively charge 5*N*log2(N)
    # real operations for each of 3 forward and 9 inverse full complex FFTs.
    fft_ops = 12 * 5 * (256 * 256) * math.log2(256 * 256)
    result["conv_gmacs"] = macs / 1e9
    result["gflops_with_fft_estimate"] = (2 * macs + fft_ops) / 1e9
    result["flop_convention"] = "2 FLOPs/MAC; thop ops plus conservative FFT estimate; elementwise ops excluded"
    assert result["gflops_with_fft_estimate"] < 10
    with torch.no_grad():
        assert model(odd[:1]).shape == odd[:1].shape
        identity = torch.nn.Identity()
        assert torch.allclose(tiled_forward(identity, x, tile=96, overlap=16), x, atol=1e-6)
    result["forward_backward_routing_checks"] = True
    if device == "cuda":
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            amp_out = model(odd[:1])
            amp_loss = amp_out.square().mean()
        amp_loss.backward()
        assert torch.isfinite(amp_out).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        result["amp_odd_size_check"] = True

    # Exercise the shared entry points for all models on both degradation modes.
    for name in ("dncnn", "ffdnet", "fcsg"):
        small = build_model(name).to(device)
        pred = predict(small, batch["lr"].to(device), batch["noise_sigma"].to(device) / 255)
        charbonnier(pred, batch["hr"].to(device)).backward()
        assert all(math.isfinite(v) for v in validate(small, batch, 25 / 255, device))
        assert all(math.isfinite(v) for v in validate(small, {"hr": batch["hr"]}, 25 / 255, device, clip=False))
        del small, pred

    # 200 updates on one fixed, real image and one fixed composite degradation.
    model = FCSGNet().to(device)
    target, noisy = hr[None].to(device), a[None].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    with torch.no_grad():
        initial = (model(noisy) - target).square().mean().item()
    losses = []
    t0 = time.perf_counter()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(noisy) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"overfit {step + 1}/200 mse={loss.item():.6f}", flush=True)
    with torch.no_grad():
        expected = model(noisy)
        final = (expected - target).square().mean().item()
    result["overfit"] = {"steps": 200, "patch": 128, "initial_mse": initial,
                         "final_mse": final, "ratio": final / initial,
                         "seconds": time.perf_counter() - t0, "losses": losses}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt_0000200.pt"
        save_checkpoint(path, 200, model, optimizer)
        restored = FCSGNet().to(device)
        opt = torch.optim.AdamW(restored.parameters())
        assert load_checkpoint(path, restored, opt, device=device) == 200
        with torch.no_grad():
            assert torch.equal(restored(noisy), expected)
        assert len(opt.state) == len(optimizer.state)
    result["checkpoint_resume_check"] = True
    result["overfit_passed"] = final < initial * 0.25 and final < 0.01
    result["passed"] = result["overfit_passed"] and device == "cuda"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "overfit"}, indent=2))
    print(f"overfit MSE {initial:.6f} -> {final:.6f}")
    assert result["overfit_passed"], "200-step overfit did not reduce MSE sufficiently"


if __name__ == "__main__":
    main()
