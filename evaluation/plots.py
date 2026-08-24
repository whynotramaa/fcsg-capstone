"""Training curves from the CSV log.

    python evaluation/plots.py --csv /kaggle/working/ckpt/train_log.csv --out results/figures

The log is sparse: loss rows carry no PSNR and validation rows carry no loss, so
each series is filtered independently rather than read as a table.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def series(rows, key):
    return [(int(r["step"]), float(r[key])) for r in rows if r.get(key)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loss = series(rows, "loss")
    if loss:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(*zip(*loss), lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("Charbonnier loss")
        ax.set_title("Training loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "loss_curve.png", dpi=130)
        print(f"wrote {out / 'loss_curve.png'}  ({len(loss)} points, "
              f"{loss[0][1]:.4f} -> {loss[-1][1]:.4f})")

    p_out = series(rows, "psnr_out")
    p_in = series(rows, "psnr_in")
    if p_out:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(*zip(*p_out), marker="o", ms=3, label="restored")
        if p_in:
            # the do-nothing floor: without it, a PSNR number means nothing
            base = sum(v for _, v in p_in) / len(p_in)
            ax.axhline(base, ls="--", c="grey", label=f"noisy input ({base:.2f} dB)")
        ax.set_xlabel("step")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Validation PSNR")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "psnr_curve.png", dpi=130)
        print(f"wrote {out / 'psnr_curve.png'}  (best {max(v for _, v in p_out):.2f} dB)")

    if not loss and not p_out:
        raise SystemExit(f"nothing to plot: {args.csv} has no loss or psnr rows")


if __name__ == "__main__":
    main()
