"""Bundle a private, one-hour Kaggle checkpoint diagnostic."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from build_phase3 import DOCKER, ROOT, SETUP, USER, bundle, cell


def build():
    files, digest = bundle()
    slug = "fcsg-p3-diagnostics"
    write = ("FILES = " + repr(files) + "\n"
             + f"assert hashlib.sha256(json.dumps(FILES, sort_keys=True).encode()).hexdigest() == {digest!r}\n"
             + "for name, source in FILES.items():\n"
             + "    path = REPO / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source)\n"
             + "os.chdir(REPO)\n"
             + f"(WORK / 'source_manifest.json').write_text(json.dumps({{'bundle_sha256': {digest!r}, 'files': list(FILES)}}, indent=2))\n")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        previous = Path.cwd()
        try:
            exec(compile(write, "bundle-cell", "exec"),
                 {"REPO": work / "code", "WORK": work, "hashlib": hashlib, "json": json, "os": os})
            manifest = json.loads((work / "source_manifest.json").read_text())
            assert manifest == {"bundle_sha256": digest, "files": list(files)}
            assert all((work / "code" / name).read_text() == source for name, source in files.items())
        finally:
            os.chdir(previous)
    run = """subprocess.run([sys.executable, '-u', 'evaluation/diagnose_phase3.py',
                '--input', '/kaggle/input', '--out', str(WORK / 'diagnostics')], check=True)
print('Diagnostic complete. Outputs:', sorted(str(p.relative_to(WORK)) for p in WORK.rglob('*') if p.is_file()))
"""
    notebook = {"cells": [
        cell("markdown", "# Phase 3 checkpoint diagnostics\n\n"
             f"Bundle SHA-256 `{digest}`. Ten fixed validation images. "
             "Routing statistics, inference latency, tile-size sensitivity, and inference-only routing substitutions.\n"),
        cell("code", SETUP), cell("code", write), cell("code", run),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.11.0"}},
        "nbformat": 4, "nbformat_minor": 4}
    out = ROOT / "build" / "kaggle" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": ["joe1995/div2k-dataset"],
                "kernel_sources": [f"{USER}/fcsg-p3-{name}" for name in
                                   ("fcsg", "dncnn-composite", "ffdnet-blind", "ffdnet-composite")],
                "competition_sources": [], "model_sources": [],
                "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{out}: {len(files)} files, SHA-256 {digest}")


if __name__ == "__main__":
    build()
