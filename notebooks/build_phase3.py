import argparse
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "whynotramaa"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

SETUP = """from pathlib import Path
import glob, hashlib, json, os, re, shutil, subprocess, sys
import torch
assert torch.cuda.is_available(), 'Enable the GPU accelerator before running.'
print('GPU:', torch.cuda.get_device_name())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'thop==0.1.1.post2209072238', 'lpips==0.1.4'], check=True)
WORK = Path('/kaggle/working')
CKPT, RESULTS = WORK / 'ckpt', WORK / 'results'
REPO = Path('/tmp/fcsg-code')
"""

SEED = """CKPT.mkdir(exist_ok=True)
def step_of(p):
    m = re.search(r'ckpt_(\\d+)\\.pt$', str(p))
    return int(m.group(1)) if m else -1
prev = sorted(glob.glob('/kaggle/input/**/ckpt_*.pt', recursive=True), key=step_of)
if prev:
    shutil.copy(prev[-1], CKPT)
    log = Path(prev[-1]).parent / 'train_log.csv'
    if log.exists():
        shutil.copy(log, CKPT)
    print('seeded', prev[-1])
else:
    print('no previous checkpoint, fresh run')
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from fcsg_net.utils import newest_checkpoint, resolve_div2k
TRAIN, VAL = resolve_div2k('/kaggle/input')
assert TRAIN != VAL, 'attach a DIV2K dataset with separate train and validation splits'
print('train', TRAIN, 'val', VAL)
"""

TRAIN = """subprocess.run([sys.executable, 'training/build_tiles.py', '--data', str(TRAIN), '--train-dir', str(TRAIN),
                '--val-dir', str(VAL), '--out', '/tmp/tiles.npy'], check=True)
subprocess.run([sys.executable, '-u', 'training/train.py', '--config', f'configs/{CONFIG}.toml',
                '--data', str(TRAIN), '--train-dir', str(TRAIN), '--val-dir', str(VAL),
                '--tiles', '/tmp/tiles.npy', '--out', str(CKPT), '--max-hours', '10.3'], check=True)
"""

FINISH = """import tomllib
steps = tomllib.loads(Path(f'configs/{CONFIG}.toml').read_text())['steps']
latest = newest_checkpoint(CKPT)
reached = torch.load(latest, map_location='cpu', weights_only=False)['step']
for old in sorted(CKPT.glob('ckpt_*.pt'), key=step_of)[:-2]:
    old.unlink()
subprocess.run([sys.executable, 'evaluation/plots.py', '--csv', str(CKPT / 'train_log.csv'),
                '--out', str(RESULTS / 'figures')], check=True)
status = {'config': CONFIG, 'step': reached, 'steps': steps, 'finished': reached >= steps}
if reached >= steps:
    subprocess.run([sys.executable, 'evaluation/eval.py', '--ckpt', str(latest), '--data', str(VAL),
                    '--val-dir', str(VAL), '--limit', '100', '--method', CONFIG,
                    '--notes', 'phase3-div2k-composite', '--out', str(RESULTS)], check=True)
    status['evaluation'] = json.loads((RESULTS / 'evaluation.json').read_text())
(WORK / 'status.json').write_text(json.dumps(status, indent=2) + '\\n')
print(json.dumps(status, indent=2))
"""


def bundle():
    files = {}
    for folder in ("src", "models", "training", "evaluation", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in (".py", ".toml") and path.name != "report.py":
                files[str(path.relative_to(ROOT))] = path.read_text()
    return files, hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(config, resume):
    cfg = tomllib.loads((ROOT / f"configs/{config}.toml").read_text())
    ablation = "seed" in cfg and cfg["model"] == "fcsg"
    files, digest = bundle()
    slug = f"fcsg-p3-{config.replace('_', '-')}"
    write = ("FILES = " + repr(files) + "\n"
             + f"assert hashlib.sha256(json.dumps(FILES, sort_keys=True).encode()).hexdigest() == {digest!r}\n"
             + "for name, source in FILES.items():\n"
             + "    path = REPO / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source)\n"
             + f"os.chdir(REPO)\nCONFIG = {config!r}\nprint('Prepared', len(FILES), 'files for', CONFIG)\n")
    cells = [
        cell("markdown", f"# {'Phase 4 ablation' if ablation else 'Phase 3 training'}: `{config}`\n\nBundle SHA-256 `{digest}`. "
             f"Schedule: {cfg['steps']:,} steps, seed {cfg.get('seed', 'unfixed')}. "
             "Each session trains up to 10.3 hours, then saves. Push again with `--resume` "
             "to continue from this output. The final session evaluates on 100 DIV2K validation images.\n"),
        cell("code", SETUP), cell("code", write), cell("code", SEED),
        cell("code", TRAIN.replace("'10.3'", repr(str(min(10.3, cfg.get("max_hours", 10.3)))))),
        cell("code", FINISH.replace("phase3-div2k-composite", "phase4-div2k-composite") if ablation else FINISH),
    ]
    if ablation:
        cells.insert(-2, cell("code", "subprocess.run([sys.executable, '-u', 'evaluation/check_ablations.py'], check=True)\n"))
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    }, "nbformat": 4, "nbformat_minor": 4}
    out = ROOT / "build" / "kaggle" / config
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": ["joe1995/div2k-dataset"],
                "kernel_sources": [f"{USER}/{slug}"] if resume else [],
                "competition_sources": [], "model_sources": [],
                "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{out}: {slug}, {len(files)} files, SHA-256 {digest}, resume={resume}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    for config in args.configs:
        build(config, args.resume)
