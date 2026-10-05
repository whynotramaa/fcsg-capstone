"""Run M1's FFDNet benchmark, M2 checks, the D2 export, and M3 throughput on Kaggle/Colab."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def download(url, path):
    urllib.request.urlretrieve(url, path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def throughput(train, val, out, steps=1000, warmup=200, budget_hours=30):
    tiles = out / "tiles.npy"
    subprocess.run([sys.executable, str(ROOT / "training/build_tiles.py"), "--data", str(train),
                    "--train-dir", str(train), "--val-dir", str(val), "--out", str(tiles)], check=True)
    result = {"steps_measured": steps, "warmup_steps": warmup, "budget_hours": budget_hours,
              "note": "tile cache, composite degradation, batch 16; validation and checkpoint time excluded",
              "models": {}}
    for config in ("dncnn_composite", "ffdnet_composite", "fcsg"):
        run = out / "throughput" / config
        subprocess.run([sys.executable, str(ROOT / "training/train.py"),
                        "--config", str(ROOT / f"configs/{config}.toml"), "--data", str(train),
                        "--train-dir", str(train), "--val-dir", str(val), "--tiles", str(tiles),
                        "--out", str(run), "--steps", str(steps)], check=True)
        with (run / "train_log.csv").open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["loss"]]
        first = next(r for r in rows if int(r["step"]) >= warmup)
        last = rows[-1]
        rate = (int(last["step"]) - int(first["step"])) / max(int(last["secs"]) - int(first["secs"]), 1)
        result["models"][config] = {"steps_per_second": rate,
                                    "steps_in_budget": int(rate * budget_hours * 3600)}
        for ckpt in run.glob("ckpt_*.pt"):
            ckpt.unlink()
    tiles.unlink()
    (out / "throughput.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="new output directory")
    args = ap.parse_args()
    import numpy as np
    import PIL
    import torch
    from fcsg_net.degrade import snapshot
    from fcsg_net.utils import resolve_div2k

    if not torch.cuda.is_available():
        raise RuntimeError("Enable the Kaggle/Colab GPU before running milestones")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = {"python": sys.version, "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                   "numpy": np.__version__, "pillow": PIL.__version__}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    weights_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth"
    weights = download(weights_url, out / "ffdnet_color.pth")
    images = out / "cbsd68"
    images.mkdir()
    request = urllib.request.Request(
        "https://api.github.com/repos/cszn/FFDNet/contents/testsets/CBSD68",
        headers={"User-Agent": "FCSG-Net-milestone-check"})
    with urllib.request.urlopen(request, timeout=60) as response:
        files = [f for f in json.load(response) if f["name"].lower().endswith(".png")]
    if len(files) != 68:
        raise ValueError(f"expected 68 author-provided CBSD68 PNGs, found {len(files)}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: download(f["download_url"], images / f["name"]), files))
    provenance = {"weights_url": weights_url, "weights_sha256": sha256(weights),
                  "weight_source": "author pretrained; not a from-scratch reproduction",
                  "images": [{"name": f["name"], "url": f["download_url"],
                              "sha256": sha256(images / f["name"])} for f in files]}
    (out / "phase1_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "evaluation/eval.py"),
                    "--ffdnet-weights", str(weights), "--data", str(images),
                    "--val-dir", str(images), "--limit", "68", "--author-protocol",
                    "--assert-psnr", "31.21", "--out", str(out / "ffdnet"),
                    "--notes", "phase1-cbsd68-author-pretrained-unclipped"], check=True)
    train, val = resolve_div2k(args.data)
    if train == val:
        raise ValueError("milestone data must have separate training and validation splits")
    subprocess.run([sys.executable, str(ROOT / "evaluation/checks.py"),
                    "--data", str(train), "--out", str(out / "phase2_checks.json")], check=True)
    result = snapshot(train, out / "d2", count=5000)
    assert len(list((out / "d2/hr").glob("*.png"))) == 5000
    assert len(list((out / "d2/lr").glob("*.png"))) == 5000
    archive = shutil.make_archive(str(out / "d2_snapshot"), "zip", out / "d2")
    result.update(archive=Path(archive).name, archive_sha256=sha256(archive),
                  manifest_sha256=sha256(out / "d2/manifest.csv"))
    (out / "phase2_snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(throughput(train, val, out), indent=2), flush=True)
    print("FFDNet benchmark, Phase 2 checks, 5,000-pair D2 snapshot, and throughput finished.", flush=True)


if __name__ == "__main__":
    main()
