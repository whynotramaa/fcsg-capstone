"""Write appendix/per_image_rows.tex from results/phase3/per_image_scores.csv."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/phase3/per_image_scores.csv"
OUT = Path(__file__).resolve().parents[1] / "appendix/per_image_rows.tex"
ORDER = ("dncnn_composite", "ffdnet_blind", "ffdnet_composite", "fcsg")

rows = defaultdict(dict)
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        rows[r["image"]][r["method"]] = r

lines = []
for image in sorted(rows):
    scores = rows[image]
    if set(scores) != set(ORDER):
        sys.exit(f"{image}: missing methods {set(ORDER) - set(scores)}")
    psnr_in = scores[ORDER[0]]["psnr_in"]
    outs = [float(scores[m]["psnr_out"]) for m in ORDER]
    best = max(outs)
    cells = [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in outs]
    lines.append(f"{image.removesuffix('.png')} & {psnr_in} & " + " & ".join(cells) + r" \\")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {len(lines)} rows to {OUT}")
