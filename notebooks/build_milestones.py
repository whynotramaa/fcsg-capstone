"""Package current source into a self-contained Kaggle notebook. No model runs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = {}
for folder in ("src", "models", "training", "evaluation", "configs"):
    for path in sorted((ROOT / folder).rglob("*")):
        if path.suffix in (".py", ".toml") and path.name != "report.py":
            files[str(path.relative_to(ROOT))] = path.read_text()
digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [cell("markdown", """# Finish Phase 1 and Phase 2

1. Import this notebook into Kaggle.
2. Enable **GPU** and **Internet** in notebook settings.
3. Add the dataset `joe1995/div2k-dataset` under **Add Input**.
4. Run all cells. A failed assertion stops the workflow.
5. Download the evidence archive and the separate D2 snapshot from the last cell.

This notebook bundles the source files listed below. It needs no Git push or
credentials. Edit the repository files and run `python notebooks/build_milestones.py`
to rebuild the bundle after a code change.

The FFDNet benchmark uses author-pretrained weights, not a from-scratch training
run. Phase 2 includes 200 optimization steps and a 5,000-pair export, not full
model training. Expect the snapshot to take additional CPU time after the GPU checks.
The actual GPU and measured results are recorded. No gate is claimed to have
passed before this notebook runs.
"""), cell("code", """from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
DATA = Path('/kaggle/input')  # On Colab, set this to the directory containing both DIV2K splits.
REPO = BASE / 'fcsg-milestone-code'
OUT = BASE / ('fcsg-milestones-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
assert DATA.exists(), f'Attach DIV2K first: {DATA}'
import torch
assert torch.cuda.is_available(), 'Enable the GPU accelerator before running.'
print('GPU:', torch.cuda.get_device_name())
print('Results:', OUT)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'thop==0.1.1.post2209072238'], check=True)
"""), cell("markdown", f"## Bundled source\n\n{len(files)} files; SHA-256 `{digest}`.\n"),
    cell("code", "FILES = " + repr(files) + "\n"
         + f"assert hashlib.sha256(json.dumps(FILES, sort_keys=True).encode()).hexdigest() == {digest!r}\n"
         + "for name, source in FILES.items():\n"
         + "    path = REPO / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source)\n"
         + "os.chdir(REPO)\nprint('Prepared', len(FILES), 'source files in', REPO)\n"),
    cell("markdown", """## Run the benchmark, model checks, and D2 export

The FFDNet gate requires 31.21 dB ± 0.5 dB on all 68 clean CBSD68 images.
The architecture checks enforce reconstruction, routing gradients, parameter
and compute budgets, a 4 GB CUDA allocator limit, and a 200-step overfit test.
The final export contains 5,000 pairs and the parameters that generated each pair.
"""), cell("code", """subprocess.run([
    sys.executable, '-u', 'evaluation/run_milestones.py',
    '--data', str(DATA), '--out', str(OUT),
], check=True)
"""), cell("markdown", "## Collect the results\n\nDownload both archives. Keep the printed GPU identity with the results.\n"),
    cell("code", """import zipfile
from IPython.display import FileLink, display

manifest = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in FILES.items()}
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
for name in ('ffdnet/evaluation.json', 'phase2_checks.json', 'phase2_snapshot.json'):
    result = json.loads((OUT / name).read_text())
    if 'overfit' in result:
        result['overfit'].pop('losses', None)
    print(name, json.dumps(result, indent=2))
archive = OUT / 'milestone_evidence.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    paths = list(OUT.glob('*.json')) + list((OUT / 'ffdnet').rglob('*')) + [OUT / 'd2/manifest.csv']
    for path in sorted(paths):
        if path.is_file():
            z.write(path, path.relative_to(OUT))
display(FileLink(str(archive)))
display(FileLink(str(OUT / 'd2_snapshot.zip')))
print('The runtime checks passed. Preserve these results before ending the session.')
""")]
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
}, "nbformat": 4, "nbformat_minor": 4}
path = ROOT / "notebooks/phase1_phase2.ipynb"
path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {path}: {len(files)} source files, SHA-256 {digest}")
