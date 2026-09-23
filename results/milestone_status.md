# Phase 1 and Phase 2 status

Updated 2026-09-23. Implementation is prepared. New model execution and the D2
export are pending the user-run Kaggle notebook. No FFDNet or FCSG-Net result
has been inserted into `benchmark.csv` without a measurement.

| Deliverable | Status | Evidence or acceptance gate |
|---|---|---|
| DnCNN baseline | Existing run complete | `benchmark.csv`: 32.646 dB DIV2K, 31.043 dB CBSD68, clipped noise, checkpoint 286000 |
| Literature review | Written | `notes/related.md`, with primary sources and explicit limits on unverified parameter counts |
| FFDNet implementation | Prepared; runtime pending | RGB model, noise maps, shared training/evaluation paths |
| FFDNet benchmark | Pending | All 68 author CBSD68 images, sigma 25, unclipped noise; within 0.5 dB of 31.21 |
| Composite degradation | Prepared; runtime pending | Blur, optional resampling, noise, JPEG; deterministic validation and per-image metadata |
| FCSG-Net architecture | Prepared; runtime pending | Soft bands, shared separable experts, spatial top-2 routing, SE fusion, residual refinement |
| Reconstruction and frequency checks | Pending | Sum of bands within 1e-5; matching sinusoidal responses at 128 and 256 pixels |
| Resources and gradients | Pending | 256x256 forward/backward, <5M parameters, <10 GFLOPs under documented counting convention, <4 GB reserved CUDA memory |
| Single-image overfit | Pending | 200 steps on one real composite-degraded crop; MSE <0.01 and <25% of initial MSE |
| D2 snapshot | Pending | 5,000 clean/degraded PNG pairs, manifest, fixed seed 1234, SHA-256 checksums |

## Run and collect the evidence

Import `notebooks/phase1_phase2.ipynb` into Kaggle. Attach
`joe1995/div2k-dataset`, enable a GPU and Internet, and run all cells.
The bundled code is a snapshot of the local files. Rebuild it after edits with
`python notebooks/build_milestones.py`.

The run writes `ffdnet/evaluation.json`, `ffdnet/benchmark.csv`,
`phase1_provenance.json`, `phase2_checks.json`, `phase2_snapshot.json`, and
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
