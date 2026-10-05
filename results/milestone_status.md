# Phase 1 and Phase 2 status

Updated 2026-10-01. The milestone notebook ran on Kaggle (Tesla T4, torch
2.10.0+cu128) on 2026-09-23 as `whynotramaa/fcsg-phase2`, bundle SHA-256
`48ef7806...6985`. Every gate passed. The executed notebook, with its printed
outputs, is `results/milestone_run_20260923.ipynb`.

| Deliverable | Status | Evidence or acceptance gate |
|---|---|---|
| DnCNN baseline | Done | `benchmark.csv`: 32.646 dB DIV2K, 31.043 dB CBSD68, clipped noise, checkpoint 286000 |
| Literature review | Written | `notes/related.md` |
| FFDNet benchmark | Passed | 31.220 dB on all 68 CBSD68 images, target 31.21, gap 0.010 dB; author-pretrained weights |
| Composite degradation | Passed | Reproducible to the byte; 244 samples/s per CPU worker on 128x128 crops |
| Reconstruction and frequency checks | Passed | Max band-sum error 6.0e-7; matching tone responses at 128 and 256 pixels |
| Resources and gradients | Passed | 97,088 parameters; 9.39 GFLOPs at 256x256 (4.66 conv GMACs); 0.60 GB peak reserved on T4 |
| Single-image overfit | Passed | MSE 0.02609 to 0.00493 in 200 steps, ratio 0.19 |
| Checkpoint resume | Passed | Restored model reproduces outputs exactly |
| D2 snapshot | Exported | 5,000 pairs, seed 1234, archive SHA-256 `271ae6c5...e4e4`, manifest SHA-256 `7ce7f310...0514` |
| Phase 3 throughput | Measured 2026-09-25 | T4, tiles, composite, batch 16: DnCNN 9.41, FFDNet 26.7, FCSG-Net 5.84 steps/s; `results/milestone_run_20260925/` |

A second run on 2026-09-25 (bundle `c73ea152...0d6a`) repeated every gate with
the Phase 3 source changes: FFDNet 31.220 dB, overfit ratio 0.18, same budgets.
Its JSON evidence is in `results/milestone_run_20260925/`. Kaggle ran Python
3.12.13, torch 2.10.0+cu128.

## Phase 3 schedule

All four composite configs train for 200,000 steps, the plan's quota fallback.
At 300,000 steps FCSG-Net alone needs 14.3 GPU-hours and the four runs exceed a
week of quota; at 200,000 each run fits one 10.3-hour session and the total is
about 22 GPU-hours. Every model keeps the same schedule, so the comparison stays
matched.

The D2 archive and evidence zip were written to that session's
`/kaggle/working`; download them from the kernel's version output.

FCSG-Net uses 2% of the 5M parameter budget and 94% of the 10 GFLOP budget.
Compute, not parameters, is the binding constraint. Report PSNR against GFLOPs:
at 256x256, DnCNN costs about 73 GFLOPs and FFDNet about 28.

## Phase 3 results

All four runs finished 200,000 steps on a Kaggle T4. Scores use 100 DIV2K
validation images, composite degradation, tiled 256 with 32 pixel overlap.
Input PSNR is 20.99 dB. Logs, figures and `evaluation.json` for each run are
in `results/phase3/<config>/`.

| Model | PSNR | SSIM | LPIPS | GFLOPs | Params | Training time |
|---|---|---|---|---|---|---|
| DnCNN composite | 27.83 | 0.780 | 0.324 | 73.4 | 558,403 | 6.7 h |
| FCSG-Net | 27.24 | 0.753 | 0.396 | 9.39 | 97,088 | 10.1 h |
| FFDNet composite, oracle noise level | 28.24 | 0.792 | 0.316 | 27.89 | 852,108 | 2.3 h |
| FFDNet blind, fixed noise level | 28.19 | 0.790 | 0.318 | 27.89 | 852,108 | 2.0 h |

Kaggle status and outputs were checked on 2026-10-01. Every submitted notebook
contains the same 21 source/config files as the local checkout, bundle SHA-256
`85d6e16f...22dd`. `phase3/source_manifest.json` records the source and result
hashes. Phase 3's comparison-table exit test is complete.

FCSG-Net trails DnCNN by 0.59 dB at 13% of its reported FLOPs. It trails blind
FFDNet by 0.95 dB at 34% of its FLOPs and 11% of its parameters. Both FFDNet
variants score higher on all 100 images in the rounded console measurements,
saved in `phase3/per_image_scores.csv`. Training takes FCSG-Net 10.1 hours
versus blind FFDNet's 2.0 hours. Fewer FLOPs do not establish a latency advantage.
The 0.05 dB oracle-versus-blind difference comes from one run of each variant;
it does not establish a repeatable benefit from noise-level conditioning.

The dense probabilities remain near uniform. Their maximum summed entropy
across three bands is 3.2958, versus 3.2782 at step 50 and 3.2847 at step
200,000, 99.66% of maximum. This log measures `dense_routing`, before top-2
selection. It cannot establish constant expert selections or absent spatial
specialization. Actual selected experts and sparse weights still need analysis.

The entropy bonus reaches zero only at the final training step, when the cosine
learning rate also reaches its minimum. At step 180,000 its coefficient is still
0.001 and the learning rate is about 5.87e-6. This is a plausible reason for
weak specialization, not a demonstrated cause. Do not change the original run.

## Next experiments

1. Use the finished FCSG-Net checkpoint on Kaggle to measure dense and sparse
   routing, expert selection frequency, spatial variation, and routing versus
   degradation parameters. Measure warmed inference latency for all models on
   the same GPU, input size, and precision.
2. Evaluate the same degraded images with 128, 256, and 512 pixel tiles on a
   small fixed subset. Training validation uses 16 degraded 128-pixel crops,
   while final evaluation degrades 100 whole images before tiling. Their
   30.59 and 27.24 dB scores are different protocols, so their gap alone does
   not prove an evaluation bug or overfitting.
3. Prioritize A2, uniform fixed mixing, against a fresh 100,000-step reference
   with the same seed and schedule. Keep top-2 and all-three uniform mixing
   distinct because changing the number of active experts changes the test.
   The new comparison configs set seed 1234 for initialization and training
   sampling. The original four runs did not fix a training seed.
4. If routing diagnostics show weak specialization, test one earlier entropy
   cutoff against that reference, with the coefficient reaching zero while
   the learning rate remains useful. Keep temperature unchanged to isolate
   the schedule. Run A1, A3, and A4 after this decision.

All four validation curves plateau near the end. Resume of a finished
200,000-step run does no training, and merely extending it changes the cosine
schedule. Preserve the four completed runs and spend the next GPU time on
diagnostics and matched ablations. The supported result today is restoration
with a small parameter count and low reported FLOPs; routing benefits remain
unproven.

## Checkpoint diagnostics, 2026-10-01

The private `whynotramaa/fcsg-p3-diagnostics` kernel, version 3, completed on a
T4. All four checkpoint hashes match the downloaded Phase 3 models. Results
are in `diagnostics/`: 120 tile scores, 80 routing samples, 20 inference
interventions, and ten routing maps. The subset uses validation indices
0, 11, 22, 33, 44, 55, 66, 77, 88, and 99. These subset scores are separate
from the 100-image Phase 3 benchmark.

The router has not collapsed to constant selection. Across repeated
degradations, the 7x7 expert is selected in 91% of low-band locations, the 5x5
expert in 88% of mid-band locations, and the 3x3 expert in 99.98% of high-band
locations. Routing maps vary spatially. Mean high-band weight for the 3x3
expert correlates with noise sigma at r=0.81 after centering within each image.
This is exploratory: all degradation parameters vary and the sample has only
ten images. It does not establish a causal or independently replicated link.

Equalizing weights within the selected pair changes mean PSNR by -0.040 dB.
Replacing the pair with equal weights across all three experts changes it
by -0.372 dB. These interventions reuse the learned weights, so they cannot
establish an advantage over a model trained with fixed routing.

At 256 pixels and batch one, warmed FP32 forward latency is 19.4 ms for
FCSG-Net, 18.0 ms for DnCNN, and 6.3 ms for blind FFDNet. TF32 is disabled.
These measurements exclude image transfers and degradation. FCSG-Net is
about 3.1 times slower than blind FFDNet under this protocol despite fewer
reported FLOPs. Increasing the FCSG-Net tile size from 256 to 512 adds only
0.013 dB on the fixed subset, so evaluation keeps 256-pixel tiles.

The next pair is `fcsg_reference` and `fcsg_uniform`: 100,000 steps, seed 1234,
the same initialization and schedule, and a fresh start. The uniform variant
mixes all three experts equally and freezes the retained 681 gate parameters.
It skips gate computation. This compares learned top-2 mixing with fixed
all-three mixing, including the difference in mixing density. Each kernel
runs a forward/backward check before training and saves after at most 5.5
hours, within a six-hour session. Its final evaluation uses all 100 images.
An entropy-schedule retry is deferred because actual routing varies.
Both kernels passed their GPU checks and began training on 2026-10-01.
`phase4/jobs.json` records the submitted versions and observed progress.
The report figures in `diagnostics/figures/` plot routing correlations,
reported FLOPs against measured latency, and tile sensitivity. Each has a PNG
and an SVG version. Generate them with `evaluation/plots.py --diagnostics
results/diagnostics --out results/diagnostics/figures`.

## Run and collect the evidence

Import `notebooks/phase1_phase2.ipynb` into Kaggle. Attach
`joe1995/div2k-dataset`, enable a GPU and Internet, and run all cells.
The bundled code is a snapshot of the local files. Rebuild it after edits with
`python notebooks/build_milestones.py`.

The run writes `ffdnet/evaluation.json`, `ffdnet/benchmark.csv`,
`phase1_provenance.json`, `phase2_checks.json`, `phase2_snapshot.json`,
`throughput.json`, and
`d2_snapshot.zip` under the output directory printed by the notebook. Its last
cell links a smaller evidence archive and the separate D2 archive. A failing
benchmark or model check stops the run before the export.

FFDNet uses author-pretrained weights to verify implementation and inference.
It is not a from-scratch training reproduction. Phase 3 must train DnCNN,
FFDNet, and FCSG-Net on the same composite degradation and schedule.

## Architecture and measurement decisions

Dense 48-channel experts applied to each band at full resolution would exceed
10 GFLOPs. The implementation retains four residual blocks and kernel sizes
7, 5, and 3, but uses depthwise separable convolutions after lossless 2x pixel
rearrangement. This changes the proposed dense expert design and must be
reported with the experimental results.

The gate selects two experts per pixel. Every expert still executes, so the
measured cost includes all nine band/expert paths. No conditional-compute
speedup is claimed. Normalized frequency cutoffs preserve cycles per pixel;
they do not remove the difference between full-image and tiled context.

The profiler reports two FLOPs per thop MAC plus an FFT estimate. Small
elementwise operations are excluded. Memory is measured on the actual Kaggle
GPU with a 4 GB allocator cap. This is not an RTX 3050 hardware test.

The historical Phase 1 PDF describes the earlier DnCNN-only state. Its comparison
with a published unclipped-noise score is not a matched-protocol reproduction.
Keep that PDF as the earlier run record; use these notes for current status.
