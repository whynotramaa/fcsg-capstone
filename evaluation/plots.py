"""Training curves from the CSV log.

    python evaluation/plots.py --csv checkpoints/train_log.csv --out results/figures

The log is sparse: loss rows carry no PSNR and validation rows carry no loss, so
each series is filtered independently rather than read as a table.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAN = float("nan")


def series(rows, key):
    return [(int(r["step"]), float(r[key])) for r in rows if r.get(key)]


def xy(pts, factor=5):
    """Unzip to x/y lists, breaking the line across abnormally large step gaps.

    A resumed session whose train_log.csv was never seeded leaves a hole in the
    log. Plotted straight through, matplotlib interpolates a confident line over
    steps that were never recorded, so insert NaN and let the gap read as one.
    """
    if len(pts) < 3:
        return [p[0] for p in pts], [p[1] for p in pts]
    deltas = sorted(b[0] - a[0] for a, b in zip(pts, pts[1:]))
    med = deltas[len(deltas) // 2] or 1
    xs, ys = [pts[0][0]], [pts[0][1]]
    for a, b in zip(pts, pts[1:]):
        if b[0] - a[0] > factor * med:
            xs.append(NAN)
            ys.append(NAN)
        xs.append(b[0])
        ys.append(b[1])
    return xs, ys


def gaps(pts, factor=5):
    """Step ranges with no logged data, for annotating."""
    if len(pts) < 3:
        return []
    deltas = sorted(b[0] - a[0] for a, b in zip(pts, pts[1:]))
    med = deltas[len(deltas) // 2] or 1
    return [(a[0], b[0]) for a, b in zip(pts, pts[1:]) if b[0] - a[0] > factor * med]


def cumulative_hours(pts):
    """secs restarts at zero every session, so accumulate for total GPU time.

    Returns the cumulative series plus the steps where a new session began.
    """
    total = prev = 0.0
    out, boundaries = [], []
    for step, s in pts:
        if s < prev:
            total += prev
            boundaries.append(step)
        prev = s
        out.append((step, (total + s) / 3600))
    return out, boundaries


def save(fig, path, note=""):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  {note}")


def plot_diagnostics(root, out):
    import numpy as np

    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    rows = json.loads((root / "routing.json").read_text())
    groups = [[r for r in rows if r["image"] == image] for image in sorted({r["image"] for r in rows})]
    names = {"fcsg": "FCSG-Net", "dncnn_composite": "DnCNN", "ffdnet_blind": "FFDNet blind"}
    figures = []
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for band in range(3):
        for col, param in enumerate(("noise_sigma", "jpeg_quality")):
            ax = axes[band, col]
            for expert, kernel in enumerate((7, 5, 3)):
                xs, ys = [], []
                for group in groups:
                    x = np.array([r[param] for r in group])
                    y = np.array([r["mean_weight"][band][expert] for r in group])
                    xs.extend(x - x.mean())
                    ys.extend(y - y.mean())
                r = next(c["r"] for c in summary["routing_correlations"]
                         if c["band"] == band and c["expert"] == expert and c["parameter"] == param)
                if r is not None:
                    assert np.isclose(np.corrcoef(xs, ys)[0, 1], r), "correlation figure disagrees with diagnostic"
                label = f"{kernel}x{kernel}, r={r:.2f}" if r is not None else f"{kernel}x{kernel}, constant"
                ax.scatter(xs, ys, s=14, alpha=0.55, label=label)
            ax.set_title(f"{('Low', 'Mid', 'High')[band]} band")
            ax.set_xlabel("Noise sigma deviation, 0-255 units" if param == "noise_sigma" else "JPEG quality deviation")
            ax.set_ylabel("Mean routing weight deviation")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle(f"Routing versus degradation, {len(rows)} samples across {len(groups)} images\n"
                 "Centered within each image; all parameters vary; exploratory correlations", fontsize=12)
    figures.append((fig, "routing_correlations"))

    benchmark = Path(__file__).resolve().parent.parent / "results" / "benchmark.csv"
    with benchmark.open(newline="") as f:
        costs = {r["method"]: float(r["gflops"]) for r in csv.DictReader(f)
                 if r["method"] in names and r["steps"] == "200000"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, name in names.items():
        measurements = summary["models"][method]["latency"]
        latency = next(r["wall_ms_median"] for r in measurements if r["size"] == 256)
        ax.scatter(costs[method], latency, s=65)
        ax.annotate(f"{name}, {latency:.1f} ms", (costs[method], latency), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Reported GFLOPs at 256x256", ylabel="Median warmed forward time, ms",
           title="Tesla T4, 256x256, batch 1, FP32, TF32 disabled", xlim=(0, 95), ylim=(0, 25))
    ax.grid(alpha=0.2)
    figures.append((fig, "compute_and_latency"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, result in summary["models"].items():
        scores = result["tile_psnr"]
        sizes = sorted(map(int, scores))
        ax.plot(sizes, [scores[str(size)] - scores["256"] for size in sizes], marker="o",
                label=names.get(method, "FFDNet oracle"))
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Tile width and height, pixels", ylabel="Mean PSNR change from 256-pixel tiles, dB",
           title=f"Tile sensitivity, {len(summary['indices'])} fixed images, 32-pixel overlap", xticks=sizes)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    figures.append((fig, "tile_sensitivity"))
    for fig, name in figures:
        fig.tight_layout()
        fig.savefig(out / f"{name}.svg")
        save(fig, out / f"{name}.png")


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--diagnostics", help="directory containing summary.json and routing.json")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.diagnostics:
        plot_diagnostics(args.diagnostics, out)
        return
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    loss = series(rows, "loss")
    if loss:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(*xy(loss), lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("Training objective")
        ax.set_title("Training loss")
        ax.grid(alpha=0.3)
        save(fig, out / "loss_curve.png",
             f"({len(loss)} points, {loss[0][1]:.4f} -> {loss[-1][1]:.4f})")

    p_out = series(rows, "psnr_out")
    p_in = series(rows, "psnr_in")
    if p_out:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for lo, hi in gaps(p_out):
            ax.axvspan(lo, hi, color="0.9", zorder=0)
        ax.plot(*xy(p_out), marker="o", ms=3, label="restored")
        if p_in:
            base = sum(v for _, v in p_in) / len(p_in)
            ax.axhline(base, ls="--", c="grey", label=f"noisy input ({base:.2f} dB)")
        ax.set_xlabel("step")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Validation PSNR")
        ax.legend()
        ax.grid(alpha=0.3)
        best_step, best = max(p_out, key=lambda t: t[1])
        save(fig, out / "psnr_curve.png", f"(best {best:.2f} dB at step {best_step})")

    lr = series(rows, "lr")
    if lr:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(*xy(lr), lw=1.2, c="tab:orange")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        ax.set_title("Cosine schedule")
        ax.grid(alpha=0.3)
        save(fig, out / "lr_curve.png",
             f"({lr[0][1]:.2e} -> {lr[-1][1]:.2e}, "
             f"{lr[-1][1] / max(lr[0][1], 1e-12) * 100:.1f}% of peak at the end)")

    secs = series(rows, "secs")
    if secs:
        hours, boundaries = cumulative_hours(secs)
        # Steps advanced during unlogged sessions, so total steps over total
        # logged hours would overstate the rate. Take the median in-session slope.
        slopes = sorted((b[0] - a[0]) / ((b[1] - a[1]) * 3600)
                        for a, b in zip(hours, hours[1:]) if b[1] > a[1] and b[0] > a[0])
        rate = slopes[len(slopes) // 2] if slopes else float("nan")
        ys, xs = xy([(s, h) for s, h in hours])

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(xs, ys, lw=1.4, c="tab:green")
        for step in boundaries:
            ax.axvline(next(h for s, h in hours if s == step), ls=":", c="grey", lw=1)
        ax.set_xlabel("logged GPU time (hours)")
        ax.set_ylabel("step")
        ax.set_title(f"Throughput, {rate:.2f} steps/s (dotted = session restart)")
        ax.grid(alpha=0.3)
        missing = sum(hi - lo for lo, hi in gaps([(s, h) for s, h in hours]))
        note = f"({hours[-1][0]:,} steps, {hours[-1][1]:.1f} h logged, {rate:.2f} steps/s)"
        if missing:
            note += f"  WARNING: {missing:,} steps ran in unlogged sessions (~{missing / rate / 3600:.1f} h)"
        save(fig, out / "throughput.png", note)

    if not loss and not p_out:
        raise SystemExit(f"nothing to plot: {args.csv} has no loss or psnr rows")


if __name__ == "__main__":
    main()
