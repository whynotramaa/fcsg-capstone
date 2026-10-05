# FCSG-Net execution plan

Working plan for the B.Tech research project (CP-I), Autumn 2026-27. Written
2026-08-24, with M1 due late August and the final report due mid-November.

Execution uses Kaggle or Colab. The local NVIDIA driver is unavailable.
See `results/milestone_status.md` for implemented work and pending measurements.
The M1/M2 gates passed on 2026-09-23. `notebooks/phase1_phase2.ipynb` now also measures Phase 3 throughput.

## 0. Read this before you write code

Three things in the proposal will cause problems later. Fixing them now costs
a day. Fixing them in October costs the project.

### 0.1 The novelty claim needs narrowing

The gap table lists UNet, DnCNN, Restormer, SwinIR, FFDNet, FFC, NAFNet. It
omits the papers that are actually closest, and a panel member who knows the
field will ask about them:

- **MWCNN** (Liu et al., 2018) already uses wavelet subbands and processes
  them jointly inside a modified U-Net.
- **SFNet / FSNet** (Cui et al., ICLR 2023 and TPAMI 2024) do *frequency
  selection* for image restoration, with learned band-wise modulation. This
  is the nearest neighbour to FCSG-Net.
- **FFTformer** (Kong et al., CVPR 2023) does restoration with frequency-domain
  attention.

The project tests spatial top-k routing over distinct shared experts after
frequency decomposition, with routing compared against recorded degradation
parameters. This is a research hypothesis, not an exhaustive novelty claim.
The literature review is in `notes/related.md`. The current implementation has
sparse mixing weights but evaluates every expert; it claims no dispatch speedup.

### 0.2 Do not feed raw complex spectra to the CNN experts

The proposal says $\hat{F}_b = \sum_e w_{b,e} E_e(\hat{G}_b)$, that is, experts
consume masked Fourier coefficients directly. This is a trap:

- Convolution in the frequency domain is pointwise multiplication in the
  spatial domain. A 7x7 conv over a spectrum has no clean interpretation, so
  the "structural inductive bias" story for $E_L$ / $E_M$ / $E_H$ collapses.
- Complex tensors mean either 2-channel real/imaginary splits (phase becomes
  discontinuous and training gets unstable) or `torch.complex64`, which many
  PyTorch ops still do not support.

**Do this instead.** Mask in the frequency domain, then immediately inverse-FFT
each band back to an image:

```
x_L, x_M, x_H = ifft(mask_L * X), ifft(mask_M * X), ifft(mask_H * X)   # real images
y_b           = sum_e w[b,e] * E_e(x_b)                                # spatial CNNs
y             = fuse(y_L, y_M, y_H)                                    # SE + refine
```

The maths is unchanged (FFT is linear, so $x_L + x_M + x_H = x$ exactly), the
band decomposition story survives intact, the kernel-size argument now makes
sense, and every expert is a plain real-valued CNN you can debug. Say so in the
report: *frequency-domain band selection, spatial-domain expert processing*.

### 0.3 Global FFT breaks patch training

You will train on 128x128 patches and evaluate on larger images. Cutoffs
specified as FFT bin indices change their physical frequency with image size.

Fix: define $r_1, r_2$ as **normalised** radii in $[0, 0.5]$ cycles/pixel
(0.5 is the axis Nyquist frequency), never in absolute bin counts. This keeps
physical cutoffs consistent. It does not make tiled inference equal to a
full-image FFT: boundary conditions, spatial context, and global SE pooling
still differ. Phase 2 tests the same sinusoidal frequencies at two resolutions.

Also make the masks **soft** (a raised-cosine transition band about 0.05 wide
rather than a hard `1[rho < r1]`). Hard masks produce ringing artefacts, which
your network will then spend capacity undoing. Keep the hard-mask version as an
ablation, it makes a nice figure.

### 0.4 Three gates of three weights is not a mixture of experts

Nine scalars per image, computed from nine input statistics, is a lookup table,
not a router. It will collapse to a near-constant, the entropy term will keep it
uniform, and the ablation "with vs without gating" will show nothing.

Make the routing carry real information by conditioning it **per spatial
location**: the gate MLP consumes local band statistics from a pooled window
(say 16x16 stride 16) and emits a routing map of shape `(B, 3 bands, 3 experts,
H/16, W/16)`, upsampled before use. Now a noisy sky routes differently from a
textured roof, the heatmaps in D5 have something to show, and the gating
ablation has a real effect to measure. Cost is a few thousand extra parameters.

Keep the global-gate variant as ablation A3. If the per-pixel version wins, that
is a result. If it does not, that is also a result, and you report it.

## 1. Phase plan

Each phase names its exit test. Do not start the next phase until the current
one passes it, and write the number down in `results/`.

### Phase 1, literature and baseline (now to 31 Aug, milestone M1)

Goal: reproduce FFDNet within 0.5 dB so you have a trustworthy yardstick.

1. Read the papers in section 4, in the order given. Keep a single
   `notes/related.md` with one paragraph per paper: what it does, what it costs
   in parameters, what it does not do. This becomes the report's related-work
   section, so write it in prose, not bullets.
2. Build the data pipeline first, not the model. Unzip DIV2K (800 train,
   100 valid HR images), write `src/fcsg_net/data.py` producing random 128x128
   crops with flips and 90-degree rotations.
3. Implement DnCNN (17 layers, 64 channels, ~0.56M params) as the smoke-test
   baseline. It is 40 lines. Train on Gaussian sigma=25 additive noise.
4. Implement FFDNet with the noise-level map input.

Exit test: record DnCNN's RGB DIV2K validation result at sigma 25. The existing
clipped-noise run reached 32.646 dB; the old 29 dB target confused this setting
with grayscale BSD68. FFDNet must be within 0.5 dB of 31.21 dB on all 68 clean
CBSD68 images using the author's unclipped-noise protocol. The milestone notebook
tests the author's pretrained weights and records their SHA-256. This verifies
implementation and inference, not training from scratch. Phase 3 still requires
training every comparison method on the same composite degradation.

Why baselines first: it forces the whole training and evaluation loop into
existence against a model whose expected numbers you already know. Any bug
surfaces here, where you can attribute it, rather than in FCSG-Net, where you
will blame the architecture.

### Phase 2, degradation pipeline and architecture (1 to 15 Sep, milestone M2)

**Degradation pipeline** (`src/fcsg_net/degrade.py`). The proposal promises
composite archival degradation. Apply in physically sensible order:

```
1. Gaussian blur        sigma_blur ~ U(0.5, 2.0)          lens/motion softening
2. Downsample-upsample  scale ~ U(1.0, 2.0), bicubic      resolution loss (optional)
3. Additive noise       sigma ~ U(5, 50)/255              film grain, sensor noise
4. JPEG compression     quality ~ U(30, 95)               archival storage artefacts
```

Sample the parameters per image and **store them in the sample dict**. You need
them for the interpretability analysis: "does the gate route to $E_H$ more as
JPEG quality drops?" is a far stronger result than a bare heatmap, and it costs
you one extra field now.

Do this **on the fly in the DataLoader**, not as 5,000 pre-rendered files.
Reasons: infinite effective dataset size, no 20 GB of PNGs to upload to Kaggle,
and reproducibility comes free from seeding. For the deliverable D2, dump a
fixed 5,000-pair snapshot with a fixed seed at the end, purely so the artefact
exists and others can reproduce your exact numbers.

Caveat: JPEG encoding on CPU is slow. Benchmark it. If the DataLoader starves
the GPU, use `num_workers=4` and pre-decode the HR images to a `.npy` memmap.

**Architecture** (`models/fcsg.py`). Build in this order, testing each piece:

1. `FrequencyDecompose`: soft radial masks, normalised radii, returns three real
   images. **Test: `assert torch.allclose(x_L + x_M + x_H, x, atol=1e-5)`.**
   This single assert catches most FFT sign, shift, and normalisation bugs.
   Use `torch.fft.rfft2` with `norm="ortho"`. Build masks directly from
   `fftfreq` and `rfftfreq`, without `fftshift`, so their bins already match.
2. `Expert`: shared class, kernel size as an argument. 4 residual blocks,
   48 channels. Three shared instances at 7x7, 5x5, 3x3 process all three bands.
   Each expert uses a lossless 2x pixel rearrangement and depthwise separable
   convolutions. Dense full-resolution convolutions exceed the 10 GFLOP budget.
3. `Gate`: local mean and energy to routing logits, per section 0.4. Upsample
   the logits, select the top two experts, and softmax those logits with
   temperature `tau`. Keep dense probabilities available for Phase 3 entropy.
4. `SEFusion`: concat three band outputs, squeeze-excite, 1x1 conv to 3 channels.
5. `Refine`: 3 spatial conv layers. Predict the **residual**, output
   `x_input + refined`. Residual prediction is worth roughly 1 dB for free and
   is what DnCNN's whole contribution was.

The implemented separable experts retain 48 channels and four residual blocks.
Their parameter count and all nine band/expert computations are checked in
`evaluation/checks.py`. The old 3.6M estimate omitted the compute cost of
applying every expert to every band. The notebook records thop MACs, counts
two FLOPs per MAC, and adds a conservative FFT estimate. Small elementwise
operations are excluded from that reported convention.

Exit test: forward and backward pass on a 1x3x256x256 tensor
under 4 GB VRAM, parameter count printed and under 5M, FLOPs measured with
`fvcore` or `thop` and under 10 GFLOPs, perfect-reconstruction assert passing,
and 200 steps of overfitting on a **single image** driving loss towards zero.
The recorded overfit gate requires final MSE below 0.01 and below one quarter
of initial MSE. The notebook records its actual GPU and enforces a 4 GB PyTorch
allocation budget; a T4 measurement does not establish RTX 3050 compatibility.
The overfit test is non-negotiable: a model that cannot memorise one image has
a bug, and you will find it in minutes instead of after a 10-hour Kaggle run.

### Phase 3, training (16 Sep to 25 Oct, milestone M3)

**Loss.**

```
L = L_char + 0.05 * L_freq + lambda_ent * L_ent
```

- `L_char`: Charbonnier, $\sqrt{(\hat f - f)^2 + \varepsilon^2}$ with
  $\varepsilon = 10^{-3}$. Better than L1/L2 for restoration and standard in
  the literature.
- `L_freq`: L1 on the magnitude of the FFT of the residual. Cheap, and it is
  thematically right for a frequency-domain paper.
- `L_ent = -sum_b H(w_b)`, with `lambda_ent` applied once in the total loss.
  Minimising a negative entropy term maximises entropy, which keeps the router
  spread out and the experts alive. The sign is right, but it is easy to flip
  by accident when you write it in code, so assert that the entropy term *falls*
  as the weights approach uniform before you trust a training run.
- Anneal `lambda_ent` from 0.01 to 0 over training. This matters: a constant
  entropy bonus pushes the router towards uniform *forever*, which is the exact
  opposite of the specialisation you want to show in D5. Keep the experts alive
  early, then let the gate commit.

**Schedule.** AdamW, lr 2e-4, cosine decay to 1e-6, 300k iterations, batch 16
at 128x128, AMP (`bfloat16` on P100 is not supported, use `float16` with a
`GradScaler`; on T4 the same). Roughly 20 to 30 hours on a P100.

**Kaggle mechanics matter more than the schedule.** Sessions die at 12 hours
and the weekly quota is 30 hours, so:

- Checkpoint model, optimiser, scaler, and iteration number every 2,000 steps
  to `/kaggle/working/`, which persists as the notebook output.
- Start every run by looking for the newest checkpoint and resuming. Write this
  before the first long run, not after losing one.
- Upload DIV2K as a **Kaggle Dataset** once. Never re-download in a notebook.
- Log to a CSV in the working directory, or use Weights & Biases with the key
  in Kaggle Secrets. Do not rely on stdout, you lose it when the session ends.
- Plan for three or four sessions across two weeks, not one heroic run.

Use the RTX 3050 only for debugging, short overfit runs, and evaluation. Use
Colab for the ablations in Phase 4 to conserve the Kaggle quota.

**Baseline parity.** Train DnCNN, FFDNet, and FCSG-Net on the *same* degradation
pipeline, the same iteration count, and the same schedule. Comparing your model
against published numbers from a different degradation setting is the single
most common way undergraduate restoration papers get torn apart. FFC-ResNet is
the optional fourth baseline, drop it without guilt if time is short and say in
the report that you did.

Exit test: the comparison table in `results/benchmark.csv` is populated with
PSNR, SSIM, LPIPS, parameters, and GFLOPs for all methods on a held-out set.

### Phase 4, ablations and interpretability (26 Oct to 15 Nov, milestone M4)

Four ablations, each isolating one claim, all at reduced length (100k
iterations) since you are measuring differences, not chasing peak numbers. Say
so explicitly in the report.

| ID | Variant | Claim it tests |
|----|---------|----------------|
| A1 | Three identical experts (all 3x3) | Kernel-size specialisation matters |
| A2 | Uniform fixed weights, no gate | Learned routing beats averaging |
| A3 | Global gate instead of spatial | Spatial conditioning matters |
| A4 | Concat fusion instead of SE | Cross-band attention matters |

Optional A5, hard masks instead of soft, to justify section 0.3.

**Interpretability (D5).** Three figures:

1. Routing heatmap: for a test image, the per-expert weight map per band,
   overlaid on the image. Look for $E_H$ activating on edges.
2. Routing versus degradation: scatter of mean gate weight against the stored
   noise sigma and JPEG quality from Phase 2. This is your strongest result if
   the correlation is real.
3. Per-band output visualisation: what each band contributes to the final image.

Report honestly. If the router collapses to near-uniform, say so, show the
entropy curve, and discuss why. A negative result reported clearly is worth
more marks than a positive one that a panel can poke a hole in.

**Report.** Start writing at the beginning of Phase 4, not at the end. The
related-work section already exists from Phase 1. Figures come from `results/`.
Budget the last five days for writing alone.

## 2. Repository layout

The scaffold already has the right directories. Fill them like this:

```
src/fcsg_net/   data.py  degrade.py  metrics.py  utils.py
models/         fcsg.py  dncnn.py  ffdnet.py  blocks.py
training/       train.py  build_tiles.py   config-driven, one entry point for all models
evaluation/     eval.py  checks.py  run_milestones.py  plots.py  report.py
                ablate.py  visualize_routing.py   (Phase 4, not written yet)
configs/        dncnn.toml  ffdnet.toml  fcsg.toml   Phase 1 Gaussian and Phase 2 checks
                dncnn_composite.toml  ffdnet_composite.toml  ffdnet_blind.toml
                a1..a4.toml   (Phase 4, not written yet)
notebooks/      kaggle_train.ipynb  thin wrapper that calls training/train.py
                phase1_phase2.ipynb  bundled milestone run, rebuilt by build_milestones.py
results/        benchmark.csv  figures/
checkpoints/    gitignored
```

FFDNet runs twice in Phase 3. `ffdnet_composite.toml` gives it the recorded
noise sigma, a non-blind upper bound. `ffdnet_blind.toml` fixes the map at
27.5/255, the middle of the noise range, so it sees no more than DnCNN and
FCSG-Net do. Report both rows.

One `train.py` driven by config files, not one script per model. The ablations
then become config diffs, which is also how you keep them honest.

Add `data/`, `checkpoints/`, and `*.zip` to `.gitignore` now. The DIV2K zips are
3.7 GB and must never enter git history.

## 3. Risks and what to do about them

| Risk | Mitigation |
|------|-----------|
| Router collapses to uniform | Entropy annealing (Phase 3). If it still collapses, report it as a finding with the entropy curve, and lean the contribution on the band decomposition. |
| FCSG-Net loses to FFDNet on PSNR | Compete on the parameter/FLOP tradeoff instead, which is the actual research question in the proposal. Plot PSNR against parameters, not PSNR alone. |
| Kaggle quota runs out mid-October | Reduce to 200k iterations, and run the ablations on Colab. Decide by 10 Oct, not later. |
| DIV2K's 800 images overfit | On-the-fly degradation gives effectively unlimited pairs. Add Flickr2K only if validation PSNR plateaus while training PSNR climbs. |
| Full-image inference exceeds 4 GB VRAM | Use 256x256 tiles with 32-pixel overlap. Report this approximation and compare boundary errors against full-image inference. |

## 4. Reading list

Read the starred ones properly. Skim the rest for the related-work section.

**Foundations, read first**

1. *Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
   for Image Denoising" (DnCNN), TIP 2017.* Residual learning, your first
   baseline.
2. *Zhang et al., "FFDNet: Toward a Fast and Flexible Solution for CNN-based
   Image Denoising", TIP 2018.* Noise-level map conditioning, your second
   baseline.

**Frequency-domain restoration, the core of your related work**

3. *Liu et al., "Multi-level Wavelet-CNN for Image Restoration", CVPRW 2018.*
   The wavelet-subband ancestor of your idea.
4. Chi et al., "Fast Fourier Convolution", NeurIPS 2020. The FFC baseline.
5. *Cui et al., "Selective Frequency Network for Image Restoration", ICLR 2023,
   and "Image Restoration via Frequency Selection", TPAMI 2024.* Read these
   carefully, they are the closest prior work and you must position against
   them.
6. Kong et al., "Efficient Frequency Domain-based Transformers for High-Quality
   Image Deblurring" (FFTformer), CVPR 2023.
7. Rahaman et al., "On the Spectral Bias of Neural Networks", ICML 2019. Why
   networks learn low frequencies first, useful for your motivation section.

**Mixture of experts and gating**

8. *Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
   Mixture-of-Experts Layer", ICLR 2017.* The origin of your gating mechanism
   and of the load-balancing loss you are re-deriving as entropy regularisation.
9. Fedus et al., "Switch Transformers", JMLR 2022. Read section 2 only, for
   routing stability tricks and why top-1 routing works.
10. Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018. Your fusion module.

**Modern baselines and evaluation**

11. Chen et al., "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022.
    The efficiency bar you are implicitly claiming to approach.
12. Zamir et al., "Restormer", CVPR 2022. Cite as the heavy-attention contrast.
13. *Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric" (LPIPS), CVPR 2018.* You are reporting this metric, so
    know what it measures and why it can disagree with PSNR.
14. Agustsson and Timofte, "NTIRE 2017 Challenge on Single Image
    Super-Resolution: Dataset and Study" (DIV2K), CVPRW 2017. Cite for the data.

**Practical**

15. PyTorch `torch.fft` docs, specifically `rfft2`, `irfft2`, `fftshift`, and
    the `norm` argument. Half an hour here saves a week of sign-error debugging.
16. PyTorch AMP recipe, for `GradScaler` and `autocast` on T4 and P100.

## 5. What to do next

1. Import `notebooks/phase1_phase2.ipynb` into Kaggle.
2. Enable a GPU and Internet, then attach `joe1995/div2k-dataset`.
3. Run all cells. Fix any failed assertion before continuing.
4. Download the result archive and D2 snapshot. Record the measured gates in
   `results/milestone_status.md` before starting Phase 3 training.
