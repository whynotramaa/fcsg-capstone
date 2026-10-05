"""Build the evidence-based PDF and Markdown report without executing a model.

Run from the repository: .venv/bin/python evaluation/build_technical_report.py
Requires the existing numpy/matplotlib plus reportlab and pypdf.
"""

import csv
import hashlib
import html
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
ASSETS = OUT / "fcsg-report-assets"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(exist_ok=True)
KEYS = ["fcsg", "dncnn_composite", "ffdnet_blind", "ffdnet_composite"]
LABELS = ["FCSG-Net", "DnCNN", "FFDNet blind", "FFDNet oracle"]
SHORT = ["FCSG", "DnCNN", "FFD blind", "FFD oracle"]
PAPER = "#faf9f5"
PALETTE = ["#bd654d", "#7c8170", "#ac936d", "#758496"]
INK, MUTED, RULE = "#27251f", "#68665b", "#dedbd2"
FONT_ROOT = Path("/home/banana/.local/share/fonts")
SANS = FONT_ROOT / "Poppins/Poppins-Regular.ttf"
SANS_BOLD = FONT_ROOT / "Poppins/Poppins-SemiBold.ttf"
MONO = FONT_ROOT / "GeistMono/GeistMono-Regular.ttf"
SERIF = ASSETS / "Zarathustra-Regular.ttf"


def js(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EVAL = [js(f"results/phase3/{k}/evaluation.json") for k in KEYS]
LOGS = [rows(f"results/phase3/{k}/train_log.csv") for k in KEYS]
MANIFEST = js("results/phase3/source_manifest.json")
CHECKS = js("results/milestone_run_20260925/phase2_checks.json")
DIAG = js("results/diagnostics/summary.json")
ROUTES = js("results/diagnostics/routing.json")
TILES = rows("results/diagnostics/tiles.csv")
INTERVENTIONS = rows("results/diagnostics/interventions.csv")
LIVE = js("output/pdf/fcsg-report-assets/live_snapshot.json")
PER = rows("results/phase3/per_image_scores.csv")
IMAGES = sorted({r["image"] for r in PER})
PS = np.array([[float(next(r["psnr_out"] for r in PER if r["method"] == k and r["image"] == i))
                for i in IMAGES] for k in KEYS])
INPUT = np.array([float(next(r["psnr_in"] for r in PER if r["method"] == "fcsg" and r["image"] == i)) for i in IMAGES])
assert PS.shape == (4, 100) and len(ROUTES) == 80 and len(TILES) == 120 and len(INTERVENTIONS) == 20
for k, ev, log in zip(KEYS, EVAL, LOGS):
    assert ev["steps"] == 200000 and ev["images"] == 100 and ev["protocol"] == "tiled-256-overlap32"
    for file, field in [("evaluation.json", "evaluation_sha256"), ("train_log.csv", "train_log_sha256")]:
        assert sha(ROOT / f"results/phase3/{k}/{file}") == MANIFEST["runs"][k][field]
    assert len([r for r in log if r["psnr_out"]]) == 100
assert np.allclose(PS.mean(axis=1), [e["psnr"] for e in EVAL], atol=.005)

for path in [SANS, SANS_BOLD, MONO]:
    font_manager.fontManager.addfont(path)
plt.rcParams.update({"font.family": "Poppins", "font.size": 10.5, "axes.labelsize": 10.5,
                     "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": RULE,
                     "grid.alpha": .7, "axes.facecolor": PAPER, "figure.facecolor": PAPER,
                     "axes.linewidth": .65, "grid.linewidth": .55,
                     "xtick.major.size": 0, "ytick.major.size": 0,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def savefig(name, fig):
    for ax in fig.axes:
        for label in ax.get_xticklabels()+ax.get_yticklabels():
            label.set_fontproperties(font_manager.FontProperties(fname=str(MONO), size=9.3))
        ax.tick_params(pad=7)
        for text in ax.texts:
            text.set_fontsize(max(text.get_fontsize(),9.3))
        legend=ax.get_legend()
        if legend:
            for text in legend.get_texts():text.set_fontsize(max(text.get_fontsize(),9))
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
    fig.savefig(ASSETS / f"{name}.png", dpi=240, facecolor=PAPER)
    fig.savefig(ASSETS / f"{name}.svg", facecolor=PAPER)
    fig.savefig(ASSETS / f"{name}.pdf", facecolor=PAPER)
    plt.close(fig)


def box(ax, x, y, w, h, text, color=INK, fill=PAPER, size=9):
    ax.add_patch(Rectangle((x, y), w, h, ec=color, fc=fill, lw=.85))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=max(size,9.3), color=color)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=INK):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.1,
                "color": color, "linestyle": "--" if dashed else "-"})


def diagram(name):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    return fig, ax


def build_figures():
    fig, ax = diagram("pipeline")
    for x, y, w, h, s in [(0.1,4.6,2,1,"800 training HR\nDIV2K images"),(3,4.6,3,1,"32 fixed crops per image\n25,600 uint8 tiles"),
                           (7,4.6,2.8,1,"Random flip / rotation\n4 loader workers"),(.1,2.8,2,1,"Gaussian blur\nU(0.5, 2.0)"),
                           (2.7,2.8,2,1,"Bicubic down / up\nU(1.0, 2.0)"),(5.3,2.8,2,1,"AWGN + clipping\nU(5, 50) / 255"),
                           (7.9,2.8,2,1,"JPEG 4:4:4\ninteger quality 30-95"),(.1,.6,2.5,1,"HR target + LR sample\nrecorded degradation metadata"),
                           (3.7,.6,2.5,1,"GPU, batch 16\nAMP restoration model"),(7.3,.6,2.5,1,"Objective + AdamW\ncheckpoint and CSV")]:
        box(ax,x,y,w,h,s,size=8.5)
    arrow(ax,2.1,5.1,3,5.1);arrow(ax,6,5.1,7,5.1)
    ax.plot([8.4,8.4,1.1],[4.6,4.15,4.15],color=INK,lw=1)
    arrow(ax,1.1,4.15,1.1,3.8)
    for x in [2.1,4.7,7.3]:arrow(ax,x,3.3,x+.6,3.3)
    ax.plot([9,9,1.35],[2.8,2.2,2.2],color=INK,lw=1);arrow(ax,1.35,2.2,1.35,1.6)
    arrow(ax,2.6,1.1,3.7,1.1);arrow(ax,6.2,1.1,7.3,1.1)
    fig.tight_layout();savefig("pipeline",fig)

    fig, ax = diagram("architecture")
    box(ax,.1,3.3,1.25,1,"RGB input x\nB x 3 x H x W",size=8)
    box(ax,1.8,3.3,1.5,1,"FP32 rFFT2\nsoft radial masks\n3 x irFFT2",size=8)
    box(ax,3.8,3.3,1.2,1,"Real bands\nL, M, H",size=8)
    box(ax,5.5,2.6,2,2.1,"Shared experts\nE7, E5, E3\n48 channels\n4 residual blocks\nall 3 run on all bands",color=PALETTE[0],size=8.5)
    box(ax,3.8,5,3.7,.7,"Local mean + energy -> gate -> top-2 maps",size=8)
    box(ax,8,3.3,1.8,1,"Weighted sum\nper band / location",size=8)
    box(ax,8,1.3,1.8,1,"Concat 9 -> SE -> 3\nrefine 3 -> 16 -> 16 -> 3",size=7.8)
    for a,b in [(1.35,1.8),(3.3,3.8),(5,5.5),(7.5,8)]:arrow(ax,a,3.8,b,3.8)
    arrow(ax,4.4,4.3,4.4,5)
    ax.plot([7.5,8.9],[5.35,5.35],color=INK,lw=1);arrow(ax,8.9,5.35,8.9,4.3,dashed=True)
    arrow(ax,8.9,3.3,8.9,2.3)
    ax.add_patch(Circle((6.5,1.8),.24,ec=INK,fc=PAPER));ax.text(6.5,1.8,"+",ha="center",va="center",fontsize=12)
    arrow(ax,8,1.8,6.75,1.8);ax.plot([.75,.75,6.5],[3.3,.5,.5],color=INK,lw=1);arrow(ax,6.5,.5,6.5,1.55)
    arrow(ax,6.25,1.8,4.7,1.8);ax.text(4.65,1.8,"Restored RGB y",ha="right",va="center",fontsize=9)
    ax.text(.1,.05,"Solid lines: image tensors. Dashed line: mixing weights. Expert parameters are shared across bands.",fontsize=8)
    fig.tight_layout();savefig("architecture",fig)

    fig, axs = plt.subplots(1,2,figsize=(9,3.7),gridspec_kw={"width_ratios":[1.8,1]})
    radius=np.linspace(0,math.sqrt(.5),700)
    lp=lambda c:.5*(1+np.cos(np.pi*np.clip((radius-c)/.05+.5,0,1)))
    low,upper=lp(.1),lp(.25)
    for m,s,c in zip([low,upper-low,1-upper],["Low","Mid","High"],PALETTE):axs[0].plot(radius,m,label=s,color=c,lw=2)
    axs[0].set(xlabel="Radial frequency (cycles/pixel)",ylabel="Mask amplitude",xlim=(0,.71),ylim=(-.03,1.08));axs[0].legend(ncol=3,frameon=False);axs[0].grid()
    for r,s in [(.10,"0.10"),(.25,"0.25")]:
        axs[1].add_patch(Circle((0,0),r,fc="none",ec=INK,lw=1.3));axs[1].text(r+.01,.012,s,fontsize=8)
    axs[1].add_patch(Rectangle((-.5,-.5),1,1,fc="none",ec=INK,lw=1))
    axs[1].axhline(0,color=RULE);axs[1].axvline(0,color=RULE);axs[1].set(xlim=(-.55,.55),ylim=(-.55,.55),aspect="equal",xlabel="Horizontal frequency",ylabel="Vertical frequency",title="Centered frequency plane")
    fig.tight_layout();savefig("frequency_masks",fig)

    fig,ax=diagram("expert_gate")
    for x,w,s in [(.1,1.6,"Band RGB\nH x W x 3"),(2.15,1.7,"Pixel unshuffle 2\nH/2 x W/2 x 12"),(4.3,1.6,"1x1, 12 -> 48\nReLU"),(6.35,1.7,"4 residual blocks\nDW K + PW 1\nReLU\nDW K + PW 1"),(8.5,1.4,"1x1, 48 -> 12\nshuffle 2\nRGB + skip")]:box(ax,x,4.3,w,1.1,s,size=8)
    for a,b in [(1.7,2.15),(3.85,4.3),(5.9,6.35),(8.05,8.5)]:arrow(ax,a,4.85,b,4.85)
    ax.text(.1,3.75,"K = 7, 5, or 3. DW is depthwise convolution; PW is pointwise convolution.",fontsize=9)
    for x,w,s in [(.1,1.7,"3 bands x RGB\n9 channels"),(2.2,2,"16x16 local pooling\nmean + squared energy\n18 channels"),(4.65,1.65,"1x1 MLP\n18 -> 24 -> 9\nReLU"),(6.75,1.3,"Bilinear upsample\nlogits / tau",),(8.5,1.4,"Top 2 of 3\nsoftmax selected\nzero third")]:box(ax,x,1.65,w,1.35,s,size=7.7)
    for a,b in [(1.8,2.2),(4.2,4.65),(6.3,6.75),(8.05,8.5)]:arrow(ax,a,2.33,b,2.33)
    ax.text(.1,.8,"Dense softmax is retained for the entropy objective. Sparse maps determine the output mixture.",fontsize=9)
    fig.tight_layout();savefig("expert_gate",fig)

    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    t=np.linspace(0,200000,401);lr=1e-6+.5*(2e-4-1e-6)*(1+np.cos(np.pi*t/200000))
    axs[0].plot(t/1000,lr, color=PALETTE[0]);axs[0].set(xlabel="Training steps (thousands)",ylabel="Learning rate");axs[0].ticklabel_format(axis="y",style="sci",scilimits=(0,0));axs[0].grid()
    axs[1].plot(t/1000,.01*(1-t/200000),color=PALETTE[0]);axs[1].set(xlabel="Training steps (thousands)",ylabel="FCSG entropy coefficient");axs[1].grid()
    fig.tight_layout();savefig("schedules",fig)

    fig,ax=plt.subplots(figsize=(9,3.5));over=CHECKS["overfit"]
    ax.plot(np.arange(200),over["losses"],color=PALETTE[0]);ax.scatter([200],[over["final_mse"]],color=PALETTE[0],s=24)
    ax.axhline(.01,ls="--",color=PALETTE[2],label="Absolute gate: MSE < 0.01")
    ax.axhline(over["initial_mse"]*.25,ls=":",color=PALETTE[3],label="Relative gate: < 25% of initial MSE")
    ax.set(xlabel="Optimizer updates (0 is initial evaluation)",ylabel="Single-image MSE",ylim=(0,.028));ax.legend(frameon=False,fontsize=8);ax.grid()
    fig.tight_layout();savefig("overfit",fig)

    old=rows("checkpoints/train_log.csv");vals=[r for r in old if r["psnr_out"]]
    fig,axs=plt.subplots(2,1,figsize=(9,4.6),sharex=True)
    for field,ax,color in [("loss",axs[0],PALETTE[0]),("psnr_out",axs[1],PALETTE[3])]:
        rr=[r for r in old if r[field]];x=np.array([int(r["step"]) for r in rr])/1000;y=np.array([float(r[field]) for r in rr])
        # Preserve session rewinds and the missing interval; never connect across them.
        split=np.flatnonzero((np.diff(x)>3)|(np.diff(x)<0))+1
        for ix in np.split(np.arange(len(x)),split):ax.plot(x[ix],y[ix],color=color,lw=.8,alpha=.8)
        ax.axvspan(132.65,210.05,fc="#eef1f3");ax.grid()
    axs[0].set(ylabel="Training objective");axs[1].set(ylabel="Fixed crop PSNR (dB)",xlabel="Training step (thousands)")
    axs[0].text(171,.065,"Missing log interval",ha="center",fontsize=8)
    fig.tight_layout();savefig("gaussian_history",fig)

    fig,axs=plt.subplots(1,3,figsize=(9,3.4))
    for ax,key,title in zip(axs,["psnr","ssim","lpips"],["PSNR (higher is better)","SSIM (higher is better)","LPIPS (lower is better)"]):
        v=[e[key] for e in EVAL];ax.bar(np.arange(4),v,color=PALETTE,width=.65);ax.set_xticks(np.arange(4),["FCSG","DnCNN","Blind","Oracle"],rotation=25,ha="right");ax.set_title(title,fontsize=9)
        ax.set_ylim(0,max(v)*1.18)
        for i,y in enumerate(v):ax.text(i,y+max(v)*.025,f"{y:.3f}" if key=="psnr" else f"{y:.4f}",ha="center",fontsize=8)
        ax.grid(axis="y");ax.set_axisbelow(True)
    fig.tight_layout();savefig("benchmark",fig)

    fig,axs=plt.subplots(2,1,figsize=(9,5.0))
    for rr,label,c in zip(LOGS,LABELS,PALETTE):
        v=[r for r in rr if r["psnr_out"]];y=[float(r["psnr_out"]) for r in v]
        axs[0].plot([int(r["step"])/1000 for r in v],y,label=label,color=c,lw=1.5)
        axs[1].plot([float(r["secs"])/3600 for r in v],y,color=c,lw=1.5)
    axs[0].set(xlabel="Optimizer steps (thousands)",ylabel="Fixed crop PSNR (dB)",ylim=(24,33));axs[0].legend(ncol=4,fontsize=8,frameon=False,loc="lower right")
    axs[1].set(xlabel="Recorded training-loop hours (includes checks and checkpoint saves)",ylabel="Fixed crop PSNR (dB)",ylim=(24,33))
    for ax in axs:ax.grid()
    fig.tight_layout();savefig("convergence",fig)

    fig,axs=plt.subplots(2,2,figsize=(9,4.5),sharex=True)
    for ax,rr,label,c in zip(axs.flat,LOGS,LABELS,PALETTE):
        r=[x for x in rr if x["loss"]];x=np.array([int(z["step"])/1000 for z in r]);y=np.array([float(z["loss"]) for z in r])
        ax.plot(x[::4],y[::4],color=c,lw=.4,alpha=.28)
        # Non-overlapping means over 50 logged minibatches (2,500 optimizer steps).
        ax.plot(x.reshape(-1,50).mean(1),y.reshape(-1,50).mean(1),color=c,lw=1.5)
        ax.set_title(label);ax.set_ylabel("Total objective");ax.grid()
    for ax in axs[1]:ax.set_xlabel("Training steps (thousands)")
    fig.tight_layout();savefig("objectives",fig)
    fig,ax=plt.subplots(figsize=(9,2.5));r=[r for r in LOGS[0] if r["entropy"]]
    ax.plot([int(x["step"])/1000 for x in r],[float(x["entropy"]) for x in r],color=PALETTE[0],lw=.65)
    ax.axhline(3*np.log(3),color=PALETTE[2],ls="--",label="3 ln(3)");ax.set(xlabel="Training steps (thousands)",ylabel="Sum of dense band entropies",ylim=(3.12,3.31));ax.legend(frameon=False);ax.grid()
    fig.tight_layout();savefig("training_entropy",fig)

    fig,axs=plt.subplots(2,1,figsize=(9,5))
    for j in range(1,4):
        d=PS[0]-PS[j];axs[0].plot(np.arange(1,101),d,color=PALETTE[j],lw=.9,label=f"FCSG - {SHORT[j]}")
    axs[0].axhline(0,color=INK,lw=.7);axs[0].set(xlabel="Sorted validation image index (0801-0900)",ylabel="Paired PSNR difference (dB)");axs[0].legend(ncol=3,fontsize=8,frameon=False);axs[0].grid()
    for p,label,c in zip(PS,LABELS,PALETTE):axs[1].plot(np.arange(1,101),np.sort(255*10**(-p/20)),label=label,color=c)
    axs[1].set(xlabel="Image rank within each model (ascending error)",ylabel="Approximate RGB RMSE (0-255 units)");axs[1].grid()
    fig.tight_layout();savefig("per_image_errors",fig)

    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for j,(ev,label,c) in enumerate(zip(EVAL,LABELS,PALETTE)):
        lat=DIAG["models"][KEYS[j]]["latency"][1]["wall_ms_median"]
        axs[0].scatter(ev["gflops"],ev["psnr"],s=55,c=c);axs[0].annotate(label,(ev["gflops"],ev["psnr"]),xytext=(4,(-14 if j==2 else 8)),textcoords="offset points",fontsize=7.5)
        axs[1].scatter(lat,ev["psnr"],s=55,c=c);axs[1].annotate(label,(lat,ev["psnr"]),xytext=(4,(-14 if j==2 else 8)),textcoords="offset points",fontsize=7.5)
    axs[0].set(xlabel="Reported GFLOPs at 256x256",ylabel="100-image PSNR (dB)",xlim=(0,100),ylim=(27,28.5));axs[1].set(xlabel="Measured 256x256 forward latency (ms)",ylabel="100-image PSNR (dB)",xlim=(0,26),ylim=(27,28.5))
    for ax in axs:ax.grid()
    fig.tight_layout();savefig("compute_latency",fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for k,label,c in zip(KEYS,LABELS,PALETTE):
        ls=DIAG["models"][k]["latency"]
        axs[0].plot([x["size"] for x in ls],[x["wall_ms_median"] for x in ls],"o-",label=label,color=c)
        axs[1].plot([x["size"] for x in ls],[x["peak_allocated_bytes"]/1e6 for x in ls],"o-",color=c)
    axs[0].set(xlabel="Square input side (pixels)",ylabel="Median forward latency (ms)");axs[1].set(xlabel="Square input side (pixels)",ylabel="Peak inference allocation (MB)")
    for ax in axs:ax.set_xticks([128,256,512]);ax.grid()
    axs[0].legend(fontsize=7,frameon=False);fig.tight_layout();savefig("latency_memory",fig)

    fig,axs=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={"width_ratios":[1.4,1]})
    for k,label,c in zip(KEYS,LABELS,PALETTE):
        means=DIAG["models"][k]["tile_psnr"];v=[means[str(t)]-means["256"] for t in [128,256,512]]
        axs[0].plot([128,256,512],v,"o-",label=label,color=c)
    axs[0].axhline(0,color=INK,lw=.7);axs[0].set(xlabel="Inference tile side (pixels)",ylabel="Mean PSNR change vs tile 256 (dB)");axs[0].set_xticks([128,256,512]);axs[0].legend(fontsize=7,frameon=False);axs[0].grid()
    ax=axs[1];ax.set(xlim=(0,480),ylim=(0,300),aspect="equal");ax.axis("off")
    ax.add_patch(Rectangle((0,0),480,256,fc="none",ec=INK,lw=1))
    for x,c in [(0,PALETTE[0]),(224,PALETTE[2])]:ax.add_patch(Rectangle((x,0),256,256,fc="none",ec=c,lw=1.8))
    ax.add_patch(Rectangle((224,0),32,256,fc="#e2ecee",ec="none",zorder=-1))
    ax.text(240,278,"32-pixel overlap",ha="center",fontsize=8);ax.text(120,-22,"Tile 1",ha="center",fontsize=8);ax.text(355,-22,"Tile 2",ha="center",fontsize=8)
    fig.tight_layout();savefig("tiling",fig)

    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    for ax,field,title in zip(axs,["mean_weight","selection_fraction"],["Mean sparse mixing weight","Fraction of pixels selecting expert"]):
        m=np.mean([r[field] for r in ROUTES],axis=0)
        cmap=LinearSegmentedColormap.from_list("routing",[PAPER,"#a7573e"])
        ax.imshow(m,vmin=0,vmax=1,cmap=cmap)
        ax.set_xticks(range(3),["7x7","5x5","3x3"]);ax.set_yticks(range(3),["Low","Mid","High"]);ax.set_title(title,fontsize=9)
        for i in range(3):
            for j in range(3):ax.text(j,i,f"{m[i,j]:.3f}",ha="center",va="center",color="white" if m[i,j]>.6 else INK,fontsize=10)
    fig.tight_layout();savefig("routing_summary",fig)

    fig,axs=plt.subplots(1,2,figsize=(9,3.6),gridspec_kw={"width_ratios":[1.8,1]})
    cor=np.array([r["r"] for r in DIAG["routing_correlations"]]).reshape(9,4)
    axs[0].imshow(cor,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    axs[0].set_xticks(range(4),["Blur","Scale","Noise","JPEG"]);axs[0].set_yticks(range(9),[f"{b} / {e}x{e}" for b in ["Low","Mid","High"] for e in [7,5,3]])
    for i in range(9):
        for j in range(4):axs[0].text(j,i,f"{cor[i,j]:+.2f}",ha="center",va="center",fontsize=7,color="white" if abs(cor[i,j])>.65 else INK)
    axs[0].set_title("Within-image-centered Pearson r",fontsize=9)
    base={r["image"]:float(r["psnr_out"]) for r in TILES if r["method"]=="fcsg" and int(r["tile"])==256}
    for j,var in enumerate(["equal_selected_top2","uniform_all3"]):
        d=np.array([float(r["psnr_out"])-base[r["image"]] for r in INTERVENTIONS if r["variant"]==var]);axs[1].scatter(np.full(len(d),j),d,color=PALETTE[j],alpha=.7,s=22);axs[1].plot([j-.2,j+.2],[d.mean()]*2,color=INK,lw=2)
    axs[1].axhline(0,color=RULE);axs[1].set_xticks([0,1],["Equal selected\ntop 2","Uniform\nall 3"]);axs[1].set(ylabel="PSNR change vs learned weights (dB)",title="Inference substitutions (10 images)");axs[1].grid(axis="y")
    fig.tight_layout();savefig("correlations_interventions",fig)

    fig,ax=plt.subplots(figsize=(9,3.2))
    for key,label,c in [("reference","Learned top-2 reference",PALETTE[0]),("uniform","Trained uniform mixture",PALETTE[2])]:
        r=LIVE["runs"][key]["validation"];ax.plot([x["step"]/1000 for x in r],[x["psnr_out"] for x in r],"o--",label=label,color=c)
    ax.set(xlabel="Training steps (thousands)",ylabel="Interim fixed crop PSNR (dB)",xticks=[2,4,6]);ax.legend(frameon=False);ax.grid()
    fig.tight_layout();savefig("live_ablation",fig)

    # Reuse the recorded image panels exactly; retypeset their labels only.
    for k,name in zip(KEYS,["qualitative_fcsg","qualitative_dncnn","qualitative_blind","qualitative_oracle"]):
        original=np.asarray(Image.open(ROOT/f"results/phase3/{k}/figures/qualitative.png").convert("RGB"))
        assert original.shape==(806,1170,3)
        fig,axs=plt.subplots(2,3,figsize=(9,5.9))
        for row,(y1,y2) in enumerate([(37,397),(426,786)]):
            item=next(z for z in PER if z["method"]==k and z["image"]==IMAGES[row])
            for col,(x1,x2) in enumerate([(21,382),(405,766),(788,1149)]):
                ax=axs[row,col];ax.imshow(original[y1:y2,x1:x2],interpolation="nearest");ax.axis("off")
                ax.set_title([f"Noisy  {float(item['psnr_in']):.2f} dB",f"Restored  {float(item['psnr_out']):.2f} dB","Ground truth"][col],fontsize=11,pad=8)
        fig.tight_layout(pad=.7);savefig(name,fig)

    original=np.asarray(Image.open(ROOT/"results/diagnostics/routing_0834.png").convert("RGB"))
    assert original.shape==(1040,1560,3)
    fig,axs=plt.subplots(3,4,figsize=(9,6))
    for row,(y1,y2) in enumerate([(91,369),(417,695),(743,1020)]):
        for col,(x1,x2) in enumerate([(74,352),(452,730),(830,1108),(1209,1487)]):
            ax=axs[row,col];ax.imshow(original[y1:y2,x1:x2],interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(["Composite input","7x7 expert","5x5 expert","3x3 expert"][col],fontsize=11,pad=9)
            if col==0:ax.set_ylabel(["Low","Mid","High"][row],fontsize=11)
    fig.tight_layout(pad=.6);savefig("routing_0834",fig)


for label, path in [("Body",SANS),("BodyBold",SANS_BOLD),("BodyItalic",SANS),("Mono",MONO),("Serif",SERIF)]:
    pdfmetrics.registerFont(TTFont(label, str(path)))
pdfmetrics.registerFontFamily("Body",normal="Body",bold="BodyBold",italic="BodyItalic",boldItalic="BodyBold")
STYLE=ParagraphStyle("body",fontName="Body",fontSize=11.1,leading=17.4,textColor=colors.HexColor(INK),spaceAfter=12)
SMALL=ParagraphStyle("small",parent=STYLE,fontSize=9.2,leading=14,textColor=colors.HexColor(MUTED))
HEADING=ParagraphStyle("heading",parent=STYLE,fontName="Serif",fontSize=23,leading=28,spaceAfter=12)
TITLE=ParagraphStyle("title",parent=HEADING,fontSize=36,leading=40)


class Report:
    def __init__(self):
        self.raw=OUT/"fcsg_technical_report_2026-10-01_base.pdf"
        self.c=canvas.Canvas(str(self.raw),pagesize=(672,1008))
        self.c.setTitle("FCSG-Net: image restoration technical report")
        self.c.setAuthor("FCSG-Net project | repository evidence")
        self.w,self.page_height=672,1008;self.x=58;self.width=self.w-116
        self.body_x=self.x+24;self.body_width=self.width-48
        self.page=0;self.y=0;self.md=[];self.overlays=[];self.titles=[];self.figure_no=0

    def new(self,title,kicker="Technical report | 01 October 2026"):
        if self.page:self.footer();self.c.showPage()
        self.page+=1;self.titles.append(title)
        self.c.setFillColor(colors.HexColor(PAPER));self.c.rect(0,0,self.w,self.page_height,fill=1,stroke=0)
        if self.page==1:
            self.c.setFillColor(colors.HexColor(INK));self.c.setFont("Serif",76)
            self.c.drawCentredString(self.w/2,self.page_height-150,title)
            self.y=self.page_height-188
        else:
            para=Paragraph(title,TITLE);_,height=para.wrap(self.width,200)
            para.drawOn(self.c,self.x,self.page_height-73-height)
            self.y=self.page_height-73-height-28
        self.md.append(f"\n## {title}\n")

    def footer(self):
        self.c.setStrokeColor(colors.HexColor(RULE));self.c.setLineWidth(.45);self.c.line(self.x,49,self.w-self.x,49)
        self.c.setFont("Body",7.7);self.c.setFillColor(colors.HexColor(MUTED))
        self.c.drawString(self.x,31,"FCSG-Net / Experimental report")
        self.c.setFont("Mono",8);self.c.drawRightString(self.w-self.x,31,f"{self.page:02d}")

    def p(self,text,small=False):
        # Trusted author text uses ReportLab's minimal bold/italic markup.
        caption=text.startswith("Figure ")
        style=SMALL if small else STYLE
        x,width=(self.x,self.width) if small else (self.body_x,self.body_width)
        if self.page==1 and not caption:
            style=ParagraphStyle("cover-body",parent=style,alignment=1,fontSize=9.8 if text.startswith("<b>Evidence") else 11.8,leading=15 if text.startswith("<b>Evidence") else 18)
        para=Paragraph(text,style);_,height=para.wrap(width,1100)
        gap=12 if not small else 10
        self.check(height+gap);para.drawOn(self.c,x,self.y-height);self.y-=height+gap
        plain=re.sub(r"<link href='([^']+)'[^>]*>(.*?)</link>",r"[\2](\1)",text)
        self.md.append(plain.replace("<b>","**").replace("</b>","**").replace("<i>","*").replace("</i>","*").replace("<br/>","\n")+"\n")

    def h(self,text):
        style=HEADING
        if self.page==1:style=ParagraphStyle("cover-subtitle",parent=HEADING,fontSize=26,leading=31,alignment=1)
        self.y-=8
        para=Paragraph(text,style);_,height=para.wrap(self.width,900)
        self.check(height+12);para.drawOn(self.c,self.x,self.y-height);self.y-=height+12;self.md.append(f"\n### {text}\n")

    def check(self,height):
        assert self.y-height>=70, f"Page {self.page} ({self.titles[-1]}) overflow: bottom {self.y-height:.1f}"

    def table(self,headers,data,widths=None,size=9,padding=6):
        sty=ParagraphStyle("table",parent=SMALL,fontSize=size+.2,leading=size+3.8,textColor=colors.HexColor(INK))
        head=ParagraphStyle("table-header",parent=sty,fontName="BodyBold",fontSize=size-.2,textColor=colors.HexColor(MUTED))
        numeric=ParagraphStyle("table-number",parent=sty,fontName="Mono",fontSize=size+.2,alignment=2)
        def cell(v,header=False):
            value=str(v)
            is_number=bool(re.match(r"^[+-]?\d",value)) and len(value)<36 and not re.search(r" (at|to|images|samples|active|retained)",value)
            return Paragraph(html.escape(value),head if header else numeric if is_number else sty)
        cells=[[cell(v,True) for v in headers]]+[[cell(v) for v in row] for row in data]
        table=Table(cells,colWidths=widths or [self.width/len(headers)]*len(headers),hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                         ("LINEABOVE",(0,0),(-1,0),.7,colors.HexColor(INK)),("LINEBELOW",(0,0),(-1,0),.45,colors.HexColor(INK)),("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor(RULE)),
                         ("TOPPADDING",(0,0),(-1,-1),padding),("BOTTOMPADDING",(0,0),(-1,-1),padding),
                         ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
        _,height=table.wrap(self.width,1100);self.check(height+19);table.drawOn(self.c,self.x,self.y-height);self.y-=height+19
        self.md.append("| "+" | ".join(map(str,headers))+" |\n| "+" | ".join(["---"]*len(headers))+" |")
        self.md.extend("| "+" | ".join(map(str,row))+" |" for row in data);self.md.append("")

    def fig(self,name,height,caption,path=None):
        self.y-=10;self.check(height+10)
        if name.startswith("qualitative_") or name=="routing_0834":path=None
        if path is None:
            self.overlays.append((self.page-1,ASSETS/f"{name}.pdf",self.x,self.y-height,self.width,height))
            link=f"fcsg-report-assets/{name}.svg"
        else:
            target=ASSETS/f"{name}.png";shutil.copyfile(path,target)
            self.c.drawImage(str(target),self.x,self.y-height,width=self.width,height=height,preserveAspectRatio=True,anchor="c")
            link=f"fcsg-report-assets/{name}.png"
        self.y-=height+12;self.figure_no+=1
        self.p(f"Figure {self.figure_no}. {caption}",small=True)
        self.md.append(f"![Figure {self.figure_no}: {caption}]({link})\n")

    def equation(self,text):
        self.check(36)
        self.c.setFillColor(colors.HexColor("#eeede7"));self.c.rect(self.x,self.y-30,self.width,30,fill=1,stroke=0)
        self.c.setFont("Mono",9.3);self.c.setFillColor(colors.HexColor(INK))
        self.c.drawString(self.x+14,self.y-19,text);self.y-=36;self.md.append(f"\n```text\n{text}\n```\n")

    def finish(self):
        self.footer();self.c.save();reader=PdfReader(str(self.raw))
        for page,path,x,y,w,h in self.overlays:
            figure=PdfReader(str(path)).pages[0];fw=float(figure.mediabox.width);fh=float(figure.mediabox.height)
            scale=min(w/fw,h/fh);tx=x+(w-fw*scale)/2;ty=y+(h-fh*scale)/2
            reader.pages[page].merge_transformed_page(figure,Transformation().scale(scale).translate(tx,ty))
        writer=PdfWriter();writer.append_pages_from_reader(reader)
        writer.add_metadata({"/Title":"FCSG-Net: image restoration technical report","/Author":"FCSG-Net project","/Subject":"Image restoration experiments; evidence cutoff 2026-10-01"})
        for i,title in enumerate(self.titles):writer.add_outline_item(title,i)
        final=OUT/"fcsg_technical_report_2026-10-01.pdf"
        with final.open("wb") as f:writer.write(f)
        self.raw.unlink()
        (OUT/"fcsg_technical_report_2026-10-01.md").write_text("# FCSG-Net technical report\n\n"+"\n".join(self.md))
        return final


def build_report():
    r=Report()
    r.new("FCSG-Net")
    r.h("Technical report")
    r.p("Frequency decomposition and spatial expert routing for composite RGB image restoration.")
    r.p("<b>Reporting date:</b> 01 October 2026. Experiment records are frozen at 03:45 IST, corresponding to 30 September 2026, 22:15 UTC. The study comprises Gaussian baseline evaluation, implementation verification, four completed composite training runs, checkpoint diagnostics, and an ongoing routing ablation.")
    r.fig("architecture",260,"FCSG-Net computation graph. Three shared spatial experts process every frequency band. Learned top-2 weights combine their outputs before channel fusion and residual refinement. Sources: models/fcsg.py and models/blocks.py.")
    r.table(["FCSG-Net evaluation","Result"],[
        ["Composite restoration, 100 DIV2K images","27.243 dB PSNR | 0.7529 SSIM | 0.3961 LPIPS"],
        ["Model size and arithmetic at 256x256","97,088 parameters | 9.39 GFLOPs"],
        ["T4 inference, FP32, batch 1, 256x256","19.40 ms median forward latency"],
    ],[235,r.width-235])
    r.p("FCSG-Net satisfies the specified parameter and arithmetic budgets. Its completed checkpoint has lower PSNR and SSIM, higher LPIPS, and higher measured forward latency than the evaluated composite baselines.",small=True)

    r.new("Abstract and contents")
    r.h("Experimental summary")
    r.p("FCSG-Net decomposes an RGB input into three frequency bands and applies shared convolutional experts with 7x7, 5x5, and 3x3 kernels. A spatial gate selects two expert contributions per band and pixel. The study evaluates this architecture under a synthetic degradation model combining blur, resampling, Gaussian noise, and JPEG compression.")
    r.p("After 200,000 training steps, FCSG-Net achieves 27.242896 dB on 100 DIV2K validation images. DnCNN, blind FFDNet, and oracle FFDNet achieve 27.830859, 28.189947, and 28.239543 dB, respectively. FCSG-Net uses 82.6% fewer parameters and 87.2% fewer reported FLOPs than DnCNN. Median 256x256 forward latency is 19.40 ms for FCSG-Net, 18.04 ms for DnCNN, and 6.30 ms for blind FFDNet on Tesla T4.")
    r.p("Routing diagnostics cover ten images with eight degradation seeds per image. Spatial weights vary with image content and correlate with sampled degradation parameters. Inference substitutions reduce mean PSNR by 0.040 dB for equal weights on the selected pair and 0.372 dB for uniform weights across all experts. These results describe dependence on the trained routing policy; they do not establish the benefit of routing during training.")
    r.p("A matched, seeded reference and uniform-routing pair is in progress. Both runs target 100,000 steps. At the common 6,000-step validation, reference and uniform crop PSNR are 28.07 and 28.36 dB. Final full-image results and repeated-seed estimates are unavailable at the reporting cutoff.")
    r.table(["Pages","Contents"],[
        ["3-8","Research objectives, degradation model, network architecture, frequency decomposition, routing, and optimization"],
        ["9-14","Implementation verification, Gaussian baseline, evaluation protocols, composite results, convergence, and optimization diagnostics"],
        ["15-21","Per-image errors, qualitative comparisons, latency, memory, tiling, and routing diagnostics"],
        ["22-26","Trained ablation, implementation issues, limitations, further experiments, provenance, and references"],
        ["27-29","Per-image input and output PSNR for all 100 validation images"],
    ],[65,r.width-65])
    r.p("Final composite metrics use 100 full validation images. Training validation uses 16 fixed crops. Diagnostic results use specified subsets or inference interventions. The active ablation has interim crop metrics only.",small=True)

    r.new("Research objectives and prior work")
    r.h("Objective and scope")
    r.p("The study tests spatial expert routing after explicit frequency decomposition under limits of 5 million parameters and 10 reported GFLOPs at 256x256. The experimental objectives are to verify the implementation, compare restoration quality with convolutional baselines, and characterize the learned routing policy.")
    r.p("Inputs and outputs are RGB images at the same spatial resolution. The degradation model approximates several forms of archival image corruption through synthetic blur, resampling, noise, and compression. Evaluation on real archival photographs and human perceptual assessment has not been performed.")
    r.h("Related methods")
    r.table(["Reference","Relation to FCSG-Net"],[
        ["DnCNN [R1]","Residual convolutional baseline. The implemented network estimates a residual that is subtracted from the degraded input."],
        ["FFDNet [R2, R3]","Uses pixel rearrangement and a noise-level map. Author weights verify inference compatibility; the composite variants train from scratch."],
        ["MWCNN [R4]","Applies wavelet decomposition and reconstruction to image restoration. Explicit subband processing is established prior work."],
        ["SFNet [R5]","Uses content-dependent local frequency selection. FCSG-Net instead uses fixed radial input masks followed by spatial top-2 expert mixing."],
        ["Sparse mixture of experts [R6]","Provides a basis for sparse expert combinations. FCSG-Net uses sparse mixing weights but computes every expert output."],
        ["Squeeze-and-excitation [R7]","Provides the channel-attention mechanism used in the fusion stage."],
        ["LPIPS [R8]","Provides the perceptual-distance metric used alongside PSNR and SSIM."],
    ],[135,r.width-135])
    r.p("Published scores from these methods use different tasks, datasets, or noise protocols and are excluded from the matched composite comparison. The study evaluates a specific architecture and routing hypothesis. The available experiments do not support a state-of-the-art performance claim or an exhaustive novelty claim.")
    r.p("Project sources: plan.md and notes/related.md. Primary references appear on page 26.",small=True)

    r.new("Data and degradation model")
    r.p("DIV2K contains 800 high-resolution training images and 100 validation images [R9]. The training cache contains 32 fixed 128x128 crops from each training image, or 25,600 uint8 tiles. The NumPy memory map contains 1,258,291,200 bytes of pixel data, approximately 1.26 GB, excluding its file header. Random flips and 90-degree rotations augment the cached crops.")
    r.fig("pipeline",285,"Training pipeline. Loader workers sample degradation parameters for each augmented clean crop and return the degraded input, clean target, and parameter metadata. Sources: data.py, degrade.py, build_tiles.py, and the submitted training notebooks.")
    r.table(["Operation","Parameter distribution and implementation"],[
        ["Blur","Pillow GaussianBlur; sigma uniformly distributed over 0.5-2.0 pixels"],
        ["Resampling","Scale uniformly distributed over 1.0-2.0; bicubic downsampling followed by upsampling to the original dimensions"],
        ["Noise","Additive white Gaussian noise (AWGN); sigma uniformly distributed over 5-50 in 8-bit units; values clipped to [0,1]"],
        ["JPEG","Integer quality 30-95 inclusive; RGB with subsampling=0, corresponding to 4:4:4"],
    ],[100,r.width-100])
    r.p("Quantization to uint8 precedes Pillow blur and JPEG encoding. Each sample records blur_sigma, scale, noise_sigma, and jpeg_quality. Validation fixes the NumPy RNG seed for each image. A separate D2 export contains 5,000 deterministic clean/degraded pairs; training samples degradation online rather than sampling exclusively from this export.",small=True)

    r.new("Network architecture")
    r.fig("architecture",250,"Network data flow and tensor dependencies. Expert parameters are shared across bands. The band dimension is folded into the batch dimension before each expert forward pass.")
    r.equation("y = x + Refine(SEFusion(concat(y_L, y_M, y_H)))")
    r.equation("y_b(u,v) = sum_e w_b,e(u,v) * E_e(x_b)(u,v)")
    r.p("FrequencyDecompose returns a tensor of shape B x 3 bands x 3 channels x H x W. Three expert instances use 7x7, 5x5, and 3x3 kernels, respectively. Each expert processes all three bands. The weighted band outputs concatenate into nine channels. SEFusion maps these channels to RGB, followed by three 3x3 refinement convolutions with channel dimensions 3 -> 16 -> 16 -> 3.")
    r.table(["Module","Parameters"],[
        ["Shared 7x7 expert","39,228"],["Shared 5x5 expert","30,012"],["Shared 3x3 expert","23,868"],
        ["Spatial gate, 18 -> 24 -> 9","681"],["SE fusion and 9 -> 3 projection","96"],["Residual refinement","3,203"],["Total","97,088"],
    ],[350,r.width-350],size=8.5,padding=4)
    r.p("Kernel size specifies the expert architecture. Assigning an expert to a semantic category such as structure or texture would require separate attribution experiments. The routing analysis therefore identifies experts by kernel size.",small=True)

    r.new("Frequency decomposition")
    r.p("Frequency decomposition uses an unshifted real FFT with orthonormal normalization. fftfreq and rfftfreq define frequencies in cycles per pixel. Radial frequency is sqrt(fx^2 + fy^2). Each frequency axis has a Nyquist limit of 0.5 cycles per pixel; the two-dimensional corner radius reaches approximately 0.707.")
    r.fig("frequency_masks",245,"Raised-cosine frequency masks at cutoffs 0.10 and 0.25 cycles per pixel, with transition width 0.05. The right panel shows a centered frequency plane. Computation uses unshifted FFT ordering. Mask curves follow the implemented analytic formula.")
    r.equation("t_c(r) = clip((r - c) / 0.05 + 0.5, 0, 1)")
    r.equation("LP_c(r) = 0.5 * (1 + cos(pi * t_c(r)))")
    r.equation("M_L = LP_0.10; M_M = LP_0.25 - LP_0.10; M_H = 1 - LP_0.25")
    r.equation("x_b = irFFT2(M_b * rFFT2(x)); sum_b x_b = x")
    r.p("The three masks sum to one. The implementation check measures a maximum band-reconstruction error of 5.96e-7 in FP32. A second check verifies matching sinusoidal responses at image sizes 128 and 256. Cutoffs expressed in cycles per pixel preserve the physical frequency definition across resolutions.")
    r.p("FFT operations execute in FP32 during mixed-precision training. This avoids half-precision CUDA FFT restrictions at odd and non-power-of-two dimensions. Inverse FFT converts each masked spectrum into a real spatial image before convolutional processing.")
    r.p("Patch and whole-image inference retain different FFT boundaries, routing context, and SE pooling context. The tile-size experiment on page 19 quantifies this dependence for the selected validation subset.",small=True)

    r.new("Expert modules and spatial routing")
    r.fig("expert_gate",290,"Expert and gate operations. Experts use lossless pixel rearrangement and depthwise separable residual blocks. The gate pools RGB band statistics in 16x16 windows using ceil_mode. Replication padding handles odd image dimensions before rearrangement; outputs are cropped to the original size.")
    r.p("Pixel unshuffle maps each RGB band to 12 channels at half the original height and width. A 1x1 convolution maps 12 channels to 48. Four residual blocks each contain two depthwise convolution and pointwise convolution stages, with ReLU between the stages. A 1x1 tail maps 48 channels to 12 and adds the rearranged input. Pixel shuffle restores the RGB dimensions.")
    r.p("The gate combines three RGB bands into nine channels and pools local means and mean squares. The resulting 18 channels pass through a 1x1 MLP with dimensions 18 -> 24 -> 9 and an intermediate ReLU. Bilinear interpolation restores full-resolution logits. Temperature is 1.0. The two largest logits per band and pixel receive a softmax-normalized weight; the remaining weight is zero.")
    r.equation("w_b,e = exp(a_b,e) / sum_(j in top2) exp(a_b,j), if e in top2")
    r.p("Every expert output is computed before mixing. Three expert calls process B*3 band images, giving nine band/expert combinations. The implementation does not skip expert kernels at locations with zero mixing weights. Reported sparsity therefore concerns the combination weights rather than conditional computation.")
    r.p("Dense softmax probabilities over all three experts support entropy regularization. Near-uniform dense probabilities can still produce different selected pairs across the image when logit ranks change. Dense entropy alone is insufficient to characterize spatial selection.",small=True)

    r.new("Optimization and training protocol")
    r.table(["Setting","Completed composite runs"],[
        ["Optimizer","AdamW; weight decay 0.0"],["Learning rate","Cosine decay from 2e-4 to 1e-6"],
        ["Training","200,000 updates; batch 16; 128x128 RGB crops"],["Execution","CUDA AMP FP16 with GradScaler; 4 persistent loader workers"],
        ["Logging","Objective every 50 steps; validation and checkpoint every 2,000 steps"],
        ["Randomness","Original training seeds are unfixed; validation seeds are fixed"],
    ],[145,r.width-145],size=8.5)
    r.equation("L_char = mean(sqrt((pred - HR)^2 + 1e-6))")
    r.equation("L_freq = mean(abs(rFFT2(pred - HR, norm='ortho')))")
    r.equation("L_ent = sum_b mean_(batch,space)(sum_e p_b,e * log(p_b,e))")
    r.equation("L = L_char + 0.05 L_freq + 0.01 (1 - step/T) L_ent")
    r.p("All composite models use Charbonnier loss with epsilon 1e-3 and a frequency-loss coefficient of 0.05. Learned-routing FCSG-Net also uses negative dense entropy. Its coefficient decreases linearly from 0.01 to zero at the configured endpoint. The log records the positive entropy value, -L_ent, summed over bands.",small=True)
    r.fig("schedules",160,"Configured learning-rate and entropy-coefficient schedules for T=200,000. The seeded ablation uses the same schedule functions with T=100,000.")
    r.p("Frequency loss computes the magnitude of the FFT of the prediction residual, rather than the difference between two spectrum magnitudes. Checkpoints save model, optimizer, scaler, step, and configuration. RNG and sampler states are absent, so a resumed trajectory is not guaranteed to be bitwise identical to uninterrupted training.",small=True)

    r.new("Implementation verification")
    r.p("Milestone verification completed on 23 September and was repeated on 25 September with the Phase 3 source changes. The second run used Tesla T4, Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, NumPy 2.0.2, and Pillow 11.3.0. The following results are taken from that run's JSON records.")
    r.table(["Check","Result"],[
        ["Band reconstruction","Maximum absolute error 5.960464e-7"],["Frequency units and FFDNet noise map","Passed"],
        ["Degradation reproducibility and rate","Byte-level repeatability passed; 233.03 samples/s with one CPU worker"],
        ["Parameter limit","97,088 < 5,000,000"],["Arithmetic limit","4.664494 profiled GMACs; 9.391903 GFLOPs with the FFT estimate"],
        ["Forward and backward memory","536.06 MB allocated; 597.69 MB reserved; batch 1, 256x256 FP32, including auxiliary tensors"],
        ["Shape, gradients, routing, and odd-size AMP","Passed"],["Checkpoint restoration","Restored model output matches the saved model output exactly"],
    ],[180,r.width-180],size=8.5)
    r.fig("overfit",190,"Single-image optimization over 200 updates. MSE decreases from 0.026092 to 0.004630 in 3.64 s, giving a final-to-initial ratio of 0.17745. Acceptance requires final MSE below 0.01 and below one-quarter of initial MSE. The final point is evaluated after the last update; the loss series is recorded before each update.")
    r.p("The deterministic D2 export contains 5,000 paired crops at seed 1234 and completes in 545.05 s. Archive and manifest hashes appear on page 25. The reported memory check measures batch-1 verification and does not measure peak batch-16 training memory.",small=True)

    r.new("Gaussian denoising baseline")
    r.p("The initial RGB DnCNN experiment uses Gaussian noise at sigma 25 and clips noisy inputs to [0,1]. Evaluation uses checkpoint 286,000 from a planned 300,000-step run. The surviving training log extends to step 287,750, but no evaluation result for that later step is recorded.")
    r.fig("gaussian_history",280,"Gaussian-run objective and fixed-crop validation PSNR. The 132,650-210,050 interval contains no surviving log entries. Curves are segmented at this gap and at a checkpoint rewind to avoid interpolating unobserved training behavior.")
    r.table(["Evaluation setting","Images","Input PSNR","Output PSNR"],[
        ["CBSD68 RGB; clipped Gaussian sigma 25",68,"20.534 dB","31.043 dB"],
        ["DIV2K validation RGB; clipped Gaussian sigma 25",100,"20.698 dB","32.646 dB"],
    ],[245,45,105,r.width-395])
    r.p("The available CSV contains 4,324 rows: 4,220 objective observations and 104 crop-validation observations. Maximum recorded crop PSNR is 34.747 dB at step 280,000. The log includes a rewind from step 56,600 to 56,050 and a missing interval of 77,400 steps. Consequently, the full optimization history cannot be reconstructed.")
    r.p("The original loader decodes a high-resolution PNG for each crop. Recorded throughput is approximately 1.8 steps/s during parts of this run. Later throughput measurements use cached tiles and composite degradation. Their different conditions prevent attribution of the entire throughput difference to caching alone.")
    r.p("Historical benchmark rows contain no SSIM, LPIPS, or GFLOP measurements. Clipping also distinguishes this experiment from the published unclipped color-denoising protocol. The historical DnCNN score is therefore retained as a separate baseline setting.",small=True)

    r.new("Evaluation protocols and metrics")
    r.table(["Protocol","Data and computation","Interpretation"],[
        ["Training validation","16 fixed augmented 128x128 crops; crop and degradation seed 1234","Monitors optimization on a fixed crop sample"],
        ["Composite evaluation","100 sorted DIV2K validation images; whole-image degradation seed equals image index; tile 256, overlap 32","Final per-image means; restored values clipped to [0,1]"],
        ["Author FFDNet reproduction","All 68 clean CBSD68 images; unclipped Gaussian sigma 25; RandomState(0) reset per image; whole-image inference","Author-pretrained checkpoint; output clamped and quantized to uint8"],
        ["Checkpoint diagnostics","10 fixed validation images; original image-index seeds; 80 center-crop routing samples","Subset measurements and inference interventions"],
    ],[108,222,r.width-330],size=8.5)
    r.equation("MSE_i = mean_RGB,pixels((prediction_i - target_i)^2)")
    r.equation("PSNR_i = 10 log10(1 / MSE_i); score = mean_i(PSNR_i)")
    r.p("Dataset PSNR is the arithmetic mean of per-image RGB PSNR. Its conversion to linear error does not recover pooled pixel MSE. Per-image RMSE on page 15 is calculated separately from each rounded PSNR observation.")
    r.p("PSNR denotes peak signal-to-noise ratio. Structural similarity (SSIM) uses an 11x11 Gaussian window with sigma 1.5 and valid convolution. Constants are C1=0.01^2 and C2=0.03^2. The implementation averages channel and spatial values. Learned perceptual image patch similarity (LPIPS) uses AlexNet features [R8] with inputs scaled to [-1,1]. Lower LPIPS indicates smaller perceptual distance. The metric network is excluded from restoration-model parameter counts.")
    r.h("FFDNet protocol reproduction")
    r.p("Author-pretrained FFDNet achieves 31.219812 dB on the 68 CBSD68 images. The paper reports 31.21 dB for color denoising at sigma 25 [R2]. The absolute difference is 0.009812 dB, within the specified 0.5 dB tolerance. This comparison verifies checkpoint compatibility and inference implementation; it does not reproduce training from scratch.")
    r.p("Sources: evaluation/eval.py, src/fcsg_net/metrics.py, ffdnet_evaluation.json, phase1_provenance.json, and the author's inference implementation [R3].",small=True)

    r.new("Composite restoration results")
    r.p("Each model completes 200,000 updates on the composite degradation distribution. Final evaluation uses the same 100 DIV2K images, per-image degradation seeds, and tiled inference policy. Mean input PSNR is 20.992 dB. FFDNet variants differ in their supplied noise-level map.")
    r.table(["Model","PSNR dB","SSIM","LPIPS","GFLOPs","Params"],[[label,f"{ev['psnr']:.3f}",f"{ev['ssim']:.4f}",f"{ev['lpips']:.4f}",f"{ev['gflops']:.2f}",f"{ev['params']:,}"] for label,ev in zip(LABELS,EVAL)],
            [113,75,65,65,75,r.width-393],size=8.7)
    r.fig("benchmark",215,"Final PSNR, SSIM, and LPIPS averaged over 100 validation images. Each model represents one training run. Higher PSNR and SSIM indicate better restoration fidelity; lower LPIPS indicates smaller perceptual distance.")
    r.p("Oracle FFDNet receives the sampled AWGN sigma before JPEG compression. Blind FFDNet receives a constant map of 27.5/255 during training and evaluation. The blind variant does not estimate noise. Relative to blind FFDNet, the oracle variant gains 0.0496 dB PSNR and 0.00215 SSIM, with LPIPS lower by 0.00142.")
    r.table(["Comparison","FCSG PSNR difference","FCSG FLOP reduction"],[
        ["DnCNN","-0.588 dB","87.2%"],["Blind FFDNet","-0.947 dB","66.3%"],["Oracle FFDNet","-0.997 dB","66.3%"],
    ],[165,155,r.width-320],size=8.5)
    r.p("The original runs have unfixed training seeds and no repeated-seed observations. Differences describe these checkpoints rather than estimates of average performance across random initializations. Equal update counts do not equalize parameter capacity or GPU time.",small=True)

    r.new("Convergence and training throughput")
    r.fig("convergence",330,"PSNR on the fixed 16-crop validation sample against optimizer updates and training-loop time. Training-loop time includes validation, checkpoint writes, and data-loading delays. These curves use a different evaluation sample from the final 100-image benchmark.")
    r.table(["Model","Training hours","Final crop dB","Maximum crop dB"],[
        [label,f"{MANIFEST['runs'][k]['training_seconds']/3600:.3f}",f"{float([z for z in log if z['psnr_out']][-1]['psnr_out']):.3f}",
         f"{max(float(z['psnr_out']) for z in log if z['psnr_out']):.3f}"] for label,k,log in zip(LABELS,KEYS,LOGS)
    ],[132,112,110,r.width-354])
    r.p("The four training loops total 21.082 recorded hours. Notebook setup, tile-cache construction, final full-image evaluation, and checkpoint diagnostics are outside this total. It therefore understates total GPU-session consumption.")
    r.p("A separate 1,000-step throughput test excludes the first 200 updates and validation/checkpoint intervals. The test measures DnCNN at 9.41 steps/s, FFDNet at 26.67 steps/s, and FCSG-Net at 5.84 steps/s on T4. Late crop-validation curves approach a plateau. The observations do not establish that extending the same training configuration would close the final quality gap.",small=True)

    r.new("Optimization diagnostics")
    r.fig("objectives",230,"Total training objectives. Faint curves show subsampled minibatch observations. Solid curves average 50 consecutive logged minibatches, corresponding to 2,500 optimizer steps. Vertical scales differ across panels.")
    r.p("The FCSG-Net objective includes negative entropy with a time-dependent coefficient. Its magnitude is therefore not directly comparable with objectives that omit this term. Cross-model restoration comparisons use the validation metrics rather than the training objective.")
    r.fig("training_entropy",120,"Dense routing entropy summed over three bands during training. The maximum is 3 ln(3)=3.29584 nats. Observations use training minibatches rather than the final routing diagnostic sample.")
    r.p("Charbonnier, frequency, and entropy loss components are not logged separately. Their individual trajectories and gradient contributions cannot be recovered. The records also lack dedicated NaN counts and gradient-norm histories.")
    r.p("The completed routing diagnostic measures mean dense band entropies of 1.08898, 1.09639, and 1.09729 nats, compared with ln(3)=1.09861. Mean sparse entropies are 0.68422, 0.69131, and 0.69178 nats, compared with ln(2)=0.69315. Spatial expert-pair selection remains variable despite these high entropy values.",small=True)

    r.new("Per-image error analysis")
    r.fig("per_image_errors",320,"Paired PSNR differences for the 100 validation images and model-specific RMSE distributions. RMSE = 255*10^(-PSNR/20), using PSNR rounded to 0.01 dB. Each RMSE curve sorts its own observations; equal ranks do not necessarily correspond to the same image.")
    r.table(["FCSG comparison","Wins / ties / losses","Mean difference","Difference range"],[
        [LABELS[j],f"{int(np.sum(PS[0]>PS[j]))} / {int(np.sum(PS[0]==PS[j]))} / {int(np.sum(PS[0]<PS[j]))}",f"{np.mean(PS[0]-PS[j]):+.4f} dB",f"{np.min(PS[0]-PS[j]):+.2f} to {np.max(PS[0]-PS[j]):+.2f} dB"] for j in range(1,4)
    ],[112,130,125,r.width-367],size=8.7)
    r.p("Both FFDNet variants exceed FCSG-Net PSNR on all 100 images. Against DnCNN, FCSG-Net records six wins, one tie at displayed precision, and 93 losses. Its minimum PSNR is 17.93 dB on 0828.png, which is also the lowest-scoring image for each baseline. The current experiment does not isolate the degradation factor responsible for this case.")
    r.p("DnCNN has three images with rounded output PSNR below input PSNR. The other models have no such observations. PSNR measures mean squared error and does not independently establish human visual preference or preservation of fine detail.")
    r.p("Appendix pages 27-29 list all input and output PSNR observations. Per-image SSIM and LPIPS are included in per_image_scores.csv. Aggregate evaluation JSON retains the original numerical precision; the per-image table is reconstructed from rounded console output.",small=True)

    r.new("Qualitative results: FCSG and DnCNN")
    r.p("The first two validation images are 0801.png and 0802.png. Columns show the degraded input, restored image, and clean target. Panels display the top-left 320x320 region. The PSNR labels refer to complete images, rather than the displayed regions. Degradation seeds are identical across models.",small=True)
    r.h("FCSG-Net, 200,000 steps")
    r.fig("qualitative_fcsg",238,"FCSG-Net restoration panels. Source: results/phase3/fcsg/figures/qualitative.png.",ROOT/"results/phase3/fcsg/figures/qualitative.png")
    r.h("DnCNN, 200,000 steps")
    r.fig("qualitative_dncnn",238,"DnCNN restoration panels. Source: results/phase3/dncnn_composite/figures/qualitative.png.",ROOT/"results/phase3/dncnn_composite/figures/qualitative.png")

    r.new("Qualitative results: FFDNet")
    r.p("The FFDNet panels use the same image regions and degradation seeds as page 16. These examples supplement the 100-image quantitative evaluation. The saved evaluation output contains no local-region PSNR or difference images.",small=True)
    r.h("Blind FFDNet, constant sigma 27.5/255")
    r.fig("qualitative_blind",238,"Blind FFDNet restoration panels. Source: results/phase3/ffdnet_blind/figures/qualitative.png.",ROOT/"results/phase3/ffdnet_blind/figures/qualitative.png")
    r.h("Oracle FFDNet, sampled noise sigma")
    r.fig("qualitative_oracle",238,"Oracle FFDNet restoration panels. Source: results/phase3/ffdnet_composite/figures/qualitative.png.",ROOT/"results/phase3/ffdnet_composite/figures/qualitative.png")

    r.new("Computational cost and inference")
    r.fig("compute_latency",180,"Full-validation PSNR against reported arithmetic and measured forward latency. Latency is measured independently with a batch-1 microbenchmark. The FFDNet variants have nearly identical computational cost.")
    r.table(["Model","128 ms","256 ms","512 ms","256 peak MB"],[
        [label]+[f"{x['wall_ms_median']:.3f}" for x in DIAG['models'][k]['latency']]+[f"{DIAG['models'][k]['latency'][1]['peak_allocated_bytes']/1e6:.2f}"] for label,k in zip(LABELS,KEYS)
    ],[125,90,90,90,r.width-395],size=8.6,padding=4)
    r.fig("latency_memory",120,"Median forward latency and peak PyTorch inference allocation at three square input sizes. Memory uses decimal megabytes and batch 1. The forward/backward implementation check on page 9 uses different execution conditions.")
    r.p("The microbenchmark uses Tesla T4, torch 2.10.0+cu128, FP32 with TF32 disabled, batch 1, 20 warmup passes, and 50 timed repetitions per size. Values are synchronized wall-clock medians. GPU-event medians and wall-clock p90 values are preserved in summary.json. Image transfer, degradation, metrics, and tile assembly are excluded.",small=True)
    r.p("Arithmetic accounting treats one multiply-accumulate (MAC) as two floating-point operations (FLOPs). It adds the FCSG FFT estimate 12*5*N*log2(N), where N=256^2. Small elementwise operations are excluded. This estimate satisfies the specified 10 GFLOP limit, but does not represent all hardware costs. No operator-level profile or end-to-end deployment latency is available.",small=True)

    r.new("Tiled inference sensitivity")
    r.p("Composite evaluation uses 256x256 tiles with 32-pixel overlap and stride 224. Final tiles are anchored to image boundaries. Outputs are summed and divided by the coverage count at each pixel. Overlap blending uses uniform averaging.")
    r.fig("tiling",240,"Tile-size sensitivity with overlap fixed at 32 pixels. PSNR differences are paired against tile size 256 using identical whole-image degradations. The adjacent diagram specifies the 256-pixel tile and 32-pixel overlap geometry.")
    r.table(["Model","Tile 128 PSNR","Tile 256 PSNR","Tile 512 PSNR"],[
        [label]+[f"{DIAG['models'][k]['tile_psnr'][str(t)]:.6f}" for t in [128,256,512]] for label,k in zip(LABELS,KEYS)
    ],[145,119,119,r.width-383])
    r.p("The ten-image subset uses validation indices 0, 11, 22, 33, 44, 55, 66, 77, 88, and 99, corresponding to images 0801, 0812, 0823, 0834, 0845, 0856, 0867, 0878, 0889, and 0900. Four models and three tile sizes produce 120 measurements.")
    r.p("Increasing the tile size from 256 to 512 improves mean FCSG-Net PSNR by 0.013276 dB. DnCNN improves by 0.002423 dB; each FFDNet variant improves by approximately 0.0075 dB. The FCSG-Net change is smaller than its 0.6-1.0 dB deficit in the full composite comparison.")
    r.p("The ten-image tile-256 FCSG-Net mean is 26.658692 dB. The 100-image mean is 27.242896 dB. These different sample means describe distinct evaluation populations. Whole-image FCSG-Net inference and seam-local error measurements are unavailable.",small=True)

    r.new("Spatial routing statistics")
    r.fig("routing_summary",195,"Average sparse weights and expert-selection fractions over 80 samples. Each sample is a 256x256 center crop extracted after whole-image degradation. Selection fractions sum to two within each band because two experts are selected at each pixel.")
    r.p("Routing diagnostics use ten source images and eight degradation seeds per image. Thus, 80 observations represent ten independent image sources. The high-band 3x3 expert has selection fraction 0.999845. Mean high-band weights remain approximately 0.256 for 7x7 and 0.223 for 5x5, indicating continued contribution from both larger-kernel experts.")
    r.fig("routing_0834",230,"Spatial routing for 0834.png at image-index seed 33. Rows denote low, mid, and high bands. Columns contain the degraded center crop and expert weights for 7x7, 5x5, and 3x3 kernels. All weights use the same 0-1 color scale. Source: diagnostics/routing_0834.png.",ROOT/"results/diagnostics/routing_0834.png")
    r.p("Spatial selection changes when expert logits exchange rank. Consequently, smooth dense probabilities can yield discontinuous selected-pair boundaries. These maps characterize the gate output; they do not identify the causal contribution of an expert to a specific restored structure.",small=True)

    r.new("Routing dependence and interventions")
    r.fig("correlations_interventions",260,"Within-image-centered Pearson correlations between mean routing weights and degradation parameters, with paired inference interventions on ten images. Points show individual intervention responses; horizontal segments show their means.")
    r.p("Correlation analysis subtracts each image's eight-sample mean from the routing and degradation variables. High-band 3x3 weight correlates +0.809 with noise sigma. Low-band 5x5 weight correlates +0.798, while low-band 7x7 weight correlates -0.728. These values describe degradation dependence within the selected image sample.")
    r.p("Blur, scale, noise, and JPEG quality vary simultaneously. The analysis contains 36 exploratory correlations, without a controlled single-parameter sweep, image-level uncertainty estimates, or multiple-comparison correction. Causal disentanglement and general expert specialization cannot be inferred from these correlations.")
    r.table(["Inference intervention","Mean PSNR change","Minimum / maximum"],[
        ["Equal weights on the selected pair","-0.039582 dB","-0.13722 / +0.02765 dB"],
        ["Uniform weights across three experts","-0.371954 dB","-0.98573 / +0.16298 dB"],
    ],[218,125,r.width-343],size=8.5)
    r.p("Interventions preserve the completed checkpoint's expert weights. Equal selected-pair weighting retains learned pair selection. Uniform weighting activates all three expert contributions at every location, including previously excluded contributions. The mean losses measure dependence on the learned inference policy.")
    r.p("The trained uniform comparison on page 22 uses matched initialization and optimization settings. It tests removal of the learned routing mechanism during training, rather than substitution after training.",small=True)

    r.new("Trained routing ablation")
    r.p("At the 01 October 2026, 03:45 IST cutoff, both private Kaggle experiments remain active. The reference reaches logged step 6,000 in 1,072 training seconds. Uniform routing reaches step 6,250 in 993 seconds. Both experiments target 100,000 updates with seed 1234. Final 100-image results are not available.")
    r.fig("live_ablation",190,"Interim PSNR on 16 fixed validation crops, rounded to 0.01 dB. At the common 6,000-step observation, reference PSNR is 28.07 dB and uniform PSNR is 28.36 dB. The 0.29 dB difference describes incomplete training.")
    r.table(["Setting","Reference","Uniform"],[
        ["Expert initialization and widths","Identical","Identical"],["Spatial mixing","Learned top 2 of 3","All three weights = 1/3"],
        ["Entropy coefficient","0.01, annealed to zero","0.0"],["Gate parameters","681 trainable","681 retained, frozen, and skipped"],
        ["Stored / trainable parameters","97,088 / 97,088","97,088 / 96,407"],
        ["Schedule and seed","100,000 steps; seed 1234","100,000 steps; seed 1234"],
    ],[165,169,r.width-334],size=8.6)
    r.p("Pre-training checks verify identical initialization, default top-2 selection, uniform weight sums, expert gradients, and absent uniform-gate gradients. Keeping the frozen gate preserves the random initialization order of subsequent modules. The uniform forward pass omits gate computation.")
    r.p("The ablation removes learned routing and entropy regularization together. It therefore estimates their combined effect rather than the independent effect of entropy. The seeded 100,000-step reference is the matched comparator; the original 200,000-step run uses a different schedule. Sessions have a six-hour limit and a 5.5-hour training cutoff, followed by full validation if training completes.",small=True)

    r.new("Implementation issues and corrections")
    r.p("Recorded issues concern data throughput, incomplete logs, notebook execution, and interpretation of evaluation results. The following corrections define the experimental conditions used in this report.")
    r.table(["Issue","Resolution or experimental consequence"],[
        ["Repeated high-resolution PNG decoding","A training-only uint8 tile cache replaces per-crop PNG decoding. Degradation and augmentation remain online. Warmed throughput is measured separately."],
        ["Missing Gaussian logs and checkpoint rewind","Available observations are retained without interpolation across missing intervals. Final baseline evaluation uses checkpoint 286,000."],
        ["Clipped and unclipped Gaussian protocols","Historical clipped-noise DnCNN results remain separate from the author-protocol FFDNet result of 31.219812 dB."],
        ["Crop and full-image validation","FCSG-Net crop PSNR is 30.592 dB; final 100-image PSNR is 27.242896 dB. Each result retains its evaluation protocol."],
        ["Diagnostic version 1: undefined digest","The notebook setup variable is corrected. Execution fails before diagnostic measurements, so this version contributes no result."],
        ["Diagnostic version 2: negation after tolist()","Negation is applied to the tensor before conversion to a list. A synthetic statistics check verifies the expression; version 3 completes successfully."],
        ["FLOPs used as a proxy for latency","A direct FP32 T4 microbenchmark measures FCSG-Net as slower than DnCNN and FFDNet despite its lower arithmetic estimate."],
        ["Sparse mixing interpreted as sparse execution","All expert outputs are computed. The implementation provides sparse weights without conditional expert dispatch."],
        ["Inference changes interpreted as trained ablations","Inference interventions remain separate from the active matched reference and uniform-training pair."],
        ["Different D2 archive hashes across exports","Archive hashes are associated with their individual export runs. The paired-data manifest identifies data content independently of the ZIP archive."],
    ],[202,r.width-202],size=8.8,padding=4)
    r.p("Sources: surviving training logs, corrected evaluation and notebook sources, diagnostics version 3 job metadata, observed version 1 and version 2 execution failures, and milestone JSON records.",small=True)

    r.new("Conclusions and further experiments")
    r.h("Conclusions")
    r.p("FCSG-Net passes reconstruction, tensor-shape, gradient, mixed-precision, and checkpoint-restoration checks. It completes composite training within the selected parameter and arithmetic limits. On the evaluated T4 configuration, its final checkpoint has lower restoration quality and higher measured forward latency than the compared convolutional baselines.")
    r.p("The learned gate varies spatially and correlates with degradation parameters in the diagnostic subset. Inference substitutions reduce mean PSNR, indicating dependence on the trained mixing policy. A benefit from learned routing during training remains unresolved until the matched ablation completes.")
    r.h("Limitations")
    r.table(["Missing measurement","Effect on interpretation"],[
        ["Repeated training seeds and independent test data","Training-run variance and untouched test-set generalization are not estimated"],
        ["Real archival images and human assessment","Synthetic restoration results do not establish performance on real archival photographs"],
        ["Operator profiles and end-to-end latency","Individual runtime costs and deployment throughput are not established"],
        ["Batch-16 peak training memory and other GPUs","Batch-1 T4 verification does not establish training memory or RTX 3050 compatibility"],
        ["Trained component ablations","Hard masks, band removal, global routing, and frequency-loss removal have not been independently evaluated"],
        ["Full restored images and residual maps","Saved image panels are insufficient for complete spatial error analysis"],
    ],[228,r.width-228],size=8.6,padding=5)
    r.h("Further experiments")
    r.p("The immediate experiment is completion of the seeded routing pair, followed by source and checkpoint verification and full-image PSNR, SSIM, LPIPS, and timing analysis. The 100,000-step pair remains separate from the earlier 200,000-step comparison.")
    r.p("Additional seeds are required to estimate the stability of any routing effect. Separate entropy and selection ablations can identify which component contributes to a difference. Operator profiling is required before optimization for conditional dispatch. Independent synthetic and real-image evaluation is required to assess generalization.")
    r.p("The late validation plateau provides no direct evidence that extending the unchanged configuration would remove the observed restoration deficit.",small=True)

    r.new("Experimental provenance")
    r.p("Evaluation JSON and training-log hashes match the recorded Phase 3 manifest. Model computation runs on Kaggle; report generation reads stored numerical records and saved image panels. The accompanying evidence_index.json lists the included source and result files with SHA-256 hashes.")
    r.table(["Experiment records","Location"],[
        ["Gaussian baseline","results/benchmark.csv; checkpoints/train_log.csv"],
        ["Milestone verification","results/milestone_run_20260923.ipynb; results/milestone_run_20260925/*.json"],
        ["Composite training and evaluation","results/phase3/{config}/evaluation.json, status.json, train_log.csv, figures/"],
        ["Per-image metrics and source manifest","results/phase3/per_image_scores.csv; source_manifest.json"],
        ["Checkpoint diagnostics","results/diagnostics/summary.json, routing.json, tiles.csv, interventions.csv, job.json"],
        ["Active routing pair","results/phase4/jobs.json; fcsg-report-assets/live_snapshot.json and live_*.csv"],
    ],[180,r.width-180],size=8.4)
    r.h("Submitted source bundles")
    r.p("The four original composite experiments use the same 21-file source bundle, SHA-256 85d6e16f72ead013faf6c0a3655d400b3b8ca80f2f14693b25943db67e8022dd. This identity predates subsequent diagnostic and ablation changes.",small=True)
    r.p("Diagnostic bundle SHA-256: d473bc10b0ad0c1ed50bd7b885c3b81fe85c4dbde9d5e61f139dd24cae019102. Seeded ablation bundle SHA-256: 4617fba4429c7b7fb18f14e553c18c15dd2032b379a4d493c6dc630404e9e431. Submitted manifests define the source used for each experiment independently of later working-tree edits.",small=True)
    snap=js("results/milestone_run_20260925/phase2_snapshot.json")
    r.p(f"D2 export, 25 September: archive SHA-256 {snap['archive_sha256']}; paired-data manifest SHA-256 {snap['manifest_sha256']}. The 23 September export records a different archive hash, 271ae6c5...e4e4. Archive identity is specific to each export.",small=True)
    r.h("Evaluated checkpoints")
    for k,label in zip(KEYS,LABELS):r.p(f"{label}: {DIAG['models'][k]['checkpoint_sha256']}",small=True)
    r.p("The evidence archive contains metric files, logs, configurations, source records, manifests, and report assets. Model checkpoints and image datasets are excluded. Their identifiers remain in the provenance records.",small=True)

    r.new("References")
    refs=[
        ("R1","Zhang et al. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP, 2017.","https://arxiv.org/abs/1608.03981"),
        ("R2","Zhang, Zuo, and Zhang. FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising. IEEE TIP, 2018. Color CBSD68 sigma-25 result in Table V.","https://arxiv.org/abs/1710.04026"),
        ("R3","Zhang et al. KAIR FFDNet inference implementation and pretrained color checkpoint.","https://github.com/cszn/KAIR/blob/master/main_test_ffdnet.py"),
        ("R4","Liu et al. Multi-level Wavelet-CNN for Image Restoration. CVPR Workshops, 2018.","https://arxiv.org/abs/1805.07071"),
        ("R5","Cui et al. Selective Frequency Network for Image Restoration. ICLR, 2023. Author implementation.","https://github.com/c-yn/SFNet"),
        ("R6","Shazeer et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. 2017.","https://arxiv.org/abs/1701.06538"),
        ("R7","Hu, Shen, and Sun. Squeeze-and-Excitation Networks. CVPR, 2018.","https://arxiv.org/abs/1709.01507"),
        ("R8","Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR, 2018.","https://arxiv.org/abs/1801.03924"),
        ("R9","Agustsson and Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. DIV2K dataset.","https://data.vision.ee.ethz.ch/cvl/DIV2K/"),
    ]
    for key,text,url in refs:r.p(f"[{key}] {text}<br/><link href='{url}' color='#a7573e'>{url}</link>",small=True)
    r.h("Reproducibility notes")
    r.p("Literature establishes the methodological context. Experimental metrics are taken from the repository records identified on page 25. Original aggregate precision is retained; per-image and interim values use the precision available in console logs.",small=True)
    r.p("Single training runs do not support estimates of training-seed uncertainty. Diagnostic subsets are reported separately from full validation. Qualitative image labels use whole-image PSNR. The active ablation remains an interim result.",small=True)
    r.p("The report is generated by evaluation/build_technical_report.py from fixed evidence files, including the frozen live_snapshot.json. The accompanying source and figures allow regeneration without querying active experiments or executing restoration models.",small=True)

    for page,start in enumerate([0,34,68],1):
        r.new(f"Appendix: per-image PSNR ({page}/3)")
        r.p("DIV2K composite validation with 256-pixel tiles and 32-pixel overlap. All values are dB, rounded to 0.01 in console output. Input degradation is identical across models. Per-image SSIM and LPIPS are preserved in the evidence CSV.",small=True)
        data=[]
        for j in range(start,min(start+34,100)):
            data.append([IMAGES[j],f"{INPUT[j]:.2f}"]+[f"{PS[k,j]:.2f}" for k in range(4)])
        r.table(["Image","Input","FCSG","DnCNN","FFD blind","FFD oracle"],data,[90,65,83,83,91,r.width-412],size=8,padding=2.5)
        r.p("Source: results/phase3/per_image_scores.csv. Aggregate means appear on page 12. Equal values at displayed precision do not establish exact equality.",small=True)
    assert r.page==29,r.page
    return r.finish()


def bundle(pdf):
    sources=[ROOT/"results/benchmark.csv", ROOT/"checkpoints/train_log.csv", ROOT/"results/phase4/jobs.json"]
    for directory in ["results/phase3","results/diagnostics","results/milestone_run_20260925"]:
        sources += [p for p in (ROOT/directory).rglob("*") if p.suffix in {".json",".csv"}]
    sources += [ROOT/p for p in ["models/fcsg.py","models/blocks.py","models/dncnn.py","models/ffdnet.py", "training/train.py","training/build_tiles.py","src/fcsg_net/degrade.py","src/fcsg_net/data.py","src/fcsg_net/metrics.py","evaluation/eval.py","evaluation/diagnose_phase3.py","evaluation/check_ablations.py"]]
    sources += list((ROOT/"configs").glob("*.toml"))
    index={"report_date_ist":"2026-10-01", "live_capture_utc":LIVE["captured_at_utc"],
           "note":"Current source is included for inspection. Original submitted source hashes are in per-run manifests. No checkpoints or datasets are bundled.",
           "files":{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))}}
    (ASSETS/"evidence_index.json").write_text(json.dumps(index,indent=2)+"\n")
    with zipfile.ZipFile(OUT/"fcsg_technical_report_evidence_2026-10-01.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in [pdf, OUT/"fcsg_technical_report_2026-10-01.md",Path(__file__).resolve()]+sorted(ASSETS.iterdir()):
            z.write(path,str(path.relative_to(ROOT)))
        for path in sorted(set(sources)):z.write(path,str(path.relative_to(ROOT)))


if __name__=="__main__":
    build_figures()
    pdf=build_report()
    bundle(pdf)
    print(f"Created {pdf} ({pdf.stat().st_size:,} bytes), 29 pages; Markdown, figures, and evidence ZIP.")
