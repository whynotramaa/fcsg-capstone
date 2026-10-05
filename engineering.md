# FCSG-Net engineering

This is the background document. It explains why the code is shaped the way it
is, what each piece does, and where the traps are. It does not list commands.
For those, read `README.md`. For the research plan, the phase deadlines, and the
reading list, read `plan.md`.

Read this once before you touch the code. Most of what looks arbitrary here is a
response to a single constraint, and once you know the constraint the rest
follows.

## The constraint that shapes everything

There is no GPU on the development machine that can do a real run. Every long
job runs on Kaggle, and Kaggle imposes three limits at once:

- 30 GPU-hours per week.
- A session dies at 12 hours.
- `/kaggle/working` is a scratch disk inside a container. It persists only when
  the run finishes normally and Kaggle saves the notebook output. A session that
  gets killed takes the whole directory with it.

The third limit is the one that catches people, because the first two are
advertised and the third is not. A training run that gets killed at the 12-hour
cap has been writing checkpoints for 12 hours and you get none of them.

Everything below follows from those three facts. Training counts steps rather
than epochs so a run can stop anywhere. The loop saves state often. The run
stops itself before the cap instead of waiting to be killed. Nothing writes a
dataset path into the code, because the container is fresh every time.

## What the project is trying to prove

FCSG-Net splits an image into three frequency bands, sends each band to a small
CNN expert, and learns a router that decides how much each expert contributes at
each location. The claim is not "frequency decomposition works". MWCNN, SFNet,
and FSNet already did that. The claim is band-adaptive sparse expert routing
with interpretable gates under 5M parameters, and the routing maps are the
result, not a side effect. `plan.md` section 0.1 explains why the claim had to
be narrowed and which papers force the narrowing.

The model and composite pipeline are implemented, and their Phase 2 gates
passed on Kaggle on 2026-09-23; see `results/milestone_status.md`.
The original machinery was built against DnCNN first on purpose. DnCNN has a published number, so if the training
loop, the degradation, the metric, or the evaluation has a bug, DnCNN misses its
number and you know the bug is in the training and evaluation code rather than
in the model. Build the same machinery against a novel architecture instead, and
every bug looks like an architecture problem.

## The data path

DIV2K is 800 training images and 100 validation images at roughly 2K
resolution. `src/fcsg_net/data.py` turns them into training patches.

`PatchDataset` returns one random 128x128 crop per image, with a random flip and
a random 90-degree rotation. `__len__` is the number of images, not the number
of crops, so one pass over the dataset is 800 crops, which is 50 steps at batch
16. A 300k-step run therefore passes over the image list 6,000 times and sees a
different crop from a different corner every time. The dataset is small. The
supply of patches is not.

Two design decisions in that file are worth knowing.

**Gaussian noise is applied on the GPU.** The clean dataset returns `hr`.
For composite degradation, `DegradedDataset` wraps the clean dataset and adds
`lr`, blur sigma, resampling scale, noise sigma, and JPEG quality inside each
DataLoader worker. Validation fixes both crop and degradation seeds. FFDNet
receives the recorded noise sigma through the shared `predict` function.

**The seed argument controls reproducibility, not shuffling.** With `seed=None`
the crop for a given index is fresh on every access, which is what training
wants. With an integer seed, index `i` always produces the same crop on any
machine, which is what validation needs. Validation PSNR from a Kaggle session
in September has to be comparable to one from October, and a moving crop would
make the number drift for reasons that have nothing to do with the model.

## The model

`models/dncnn.py` is DnCNN from Zhang et al. 2017: 17 convolution layers at 64
channels, batch norm on the middle 15, ReLU throughout. It is 558,403
parameters. Run `python models/dncnn.py` to print that number and check the
forward pass.

The one line that matters is `return x - self.body(x)`. The network predicts the
noise and the forward pass subtracts it, rather than predicting the clean image
directly. That is the paper's whole contribution and it is worth roughly 1 dB.
FCSG-Net's `Refine` block does the same thing for the same reason.

The self-check at the bottom of the file asserts the parameter count is between
550,000 and 570,000. That looks pedantic until a typo in `width` or `depth`
silently gives you a different model, and you find out after a ten-hour run
produced a number that does not match anything published.

## The training loop

`training/train.py` is one entry point for every model in the project. The model
name and its arguments come from a TOML config, so the Phase 4 ablations become
config diffs rather than four forked copies of the script that drift apart.

**It counts steps, not epochs.** `for step in range(start + 1, total + 1)` over
an infinite generator wrapped around the DataLoader. Epochs would be a bad fit
here. An epoch is only 50 steps, and a run has to stop and resume at arbitrary
points, which a half-finished epoch makes awkward.

**The learning rate is computed from the step.** `cosine_lr(step, total, ...)`
is a pure function of the step number, so resuming needs no scheduler state in
the checkpoint and cannot get out of phase with the optimiser. The cost is that
`steps` is baked into the shape of the schedule. Change `steps` halfway through
a run and the learning rate jumps, because the cosine now decays to zero at a
different point. If you want to shorten the run, decide before the next session
starts, not during one.

**The loss is Charbonnier**, the smooth L1 variant with `eps=1e-3`. It is
standard for restoration and it is what FCSG-Net will use, so the Phase 3
comparison is like for like.

**Mixed precision uses float16 with a `GradScaler`, not bfloat16.** Neither the
P100 nor the T4 supports bf16, and both are what Kaggle hands out. float16 needs
the loss scaler to keep small gradients from flushing to zero, which is why
`scaler` is checkpointed alongside the optimiser.

**Validation runs on a fixed batch with a fixed noise seed.** `validate` builds
its noise from `torch.Generator().manual_seed(VAL_SEED)` over 16 seeded crops.
Same crops, same noise, every session. It reports the PSNR of the noisy input as
well as the output, and that first number is the point. It is the do-nothing
floor. A model reporting 29 dB means nothing until you know the input was 20 dB.

## Checkpoints and resuming

Three functions in `src/fcsg_net/utils.py` do all of it. `save_checkpoint`
writes the model, optimiser, scaler, step number, and the config that produced
them. `newest_checkpoint` finds the highest-numbered `ckpt_*.pt` in a directory,
sorting by the number parsed out of the filename rather than by string order.
`load_checkpoint` restores the three state dicts and returns the step to resume
from.

`train.py` calls `newest_checkpoint(out_dir)` on startup, every time, with no
flag to enable it. If a checkpoint is there, it resumes and prints `RESUMED
from ... at step N`. If not, it prints `starting from step 0`. Watch for one of
those two lines in the first few seconds of every run, because a run that
silently restarts from zero looks exactly like a run that is working.

Checkpoints are about 6.8 MB each, which is the 2.2 MB of weights plus AdamW's
two moment buffers. At `ckpt_every = 2000` a full 300k-step run leaves 150 of
them, or roughly 1 GB, comfortably inside Kaggle's output limit.

**The run stops itself.** `max_hours` in the config, default 11.0, is the wall
clock budget. When the loop passes it, it saves a checkpoint, prints how many
GPU-hours remain at the rate it just measured, and breaks. The stop exists
entirely because of the third Kaggle limit. A run that exits normally lets the
notebook finish and lets Kaggle persist `/kaggle/working`. A run still going at
12 hours gets killed and the checkpoints die in the container.

Use **Save Version, then Save and Run All (Commit)** rather than running
interactively. A committed run does not depend on your browser staying open.

**Resuming across sessions is manual, and Kaggle makes it so.** A new session
gets a fresh container, so the previous session's checkpoints have to be
attached as an input dataset. The resume cell in
`notebooks/kaggle_train.ipynb` handles it. The cell finds the highest-step
checkpoint under `/kaggle/input`, copies it and the CSV log into
`/kaggle/working/ckpt`, and lets `train.py` pick it up. It selects by parsed
step number, because sorting the paths as strings puts a version 10 output
before a version 3 output and would quietly rewind you 70,000 steps.

## Evaluation

`evaluation/eval.py` scores a checkpoint on full validation images, not on
patches. Training on 128x128 crops and reporting on 128x128 crops would flatter
the model, and the published DnCNN numbers everyone compares against are
full-image numbers.

To limit activation memory, `tiled_forward` uses 256x256 tiles with 32 pixels
of overlap and averages where they overlap. This is an approximation even for
DnCNN because tile-edge padding changes its context. FCSG-Net also changes
FFT boundaries, routing windows, and SE pooling context. Cutoffs in cycles per
pixel preserve physical frequency but do not make tiled and whole-image
inference identical. The FFDNet reproduction uses whole CBSD68 images and
keeps every pixel, including odd dimensions.

`eval.py` appends one row to `results/benchmark.csv` with the method, step
count, sigma, input and output PSNR, gain, parameters, and a free-text note.
That file is the Phase 3 comparison table. FFDNet and FCSG-Net append to the
same file rather than producing their own, which is how the comparison stays
honest. The same script produced every row from the same data. Tag throwaway
runs with `--notes smoke` so a two-minute check never gets mistaken
for a real number.

## Logging

`CSVLog` appends rows to `train_log.csv` and nothing else. stdout does not
survive a killed session. A file in `/kaggle/working` does, and the resume cell
copies it forward, so the log spans every session of a run.

The log is sparse on purpose. A loss row carries `step`, `loss`, and `lr` with
the PSNR columns blank. A validation row carries the two PSNR columns with the
loss columns blank. They arrive at different intervals, so forcing them into one
dense row would mean either logging loss only every 2,000 steps or running
validation every 50. `evaluation/plots.py` filters each series independently for
that reason, which is why it reads the file with `csv.DictReader` and a per-key
filter instead of loading a dataframe.

## Finding the images

No Kaggle dataset slug appears anywhere in the code. Public DIV2K datasets on
Kaggle use different directory layouts, they get renamed, and a Colab unzip
produces a third layout.

`resolve_div2k` takes any directory above the images. It walks down, collects
every directory that directly contains image files, then picks the training and
validation directories by matching keywords against their names in priority
order. When it cannot decide, it raises an error that lists every directory it
found with an image count, so the fix is to copy two of those paths into
`--train-dir` and `--val-dir`.

One case deserves suspicion. If the search finds exactly one image directory, it
uses it for both training and validation and prints a warning. That is fine for
a smoke run and worthless for a real one, because the validation crops then come
from images the model trains on. If you see that warning during a real run,
stop and attach a dataset that has both splits.

## What the gate checks catch

`notebooks/checks.ipynb` takes about two minutes of GPU time and it exists
because each of its five cells catches a class of bug that is expensive to find
later.

1. **Rendered crops.** Blank crops, swapped colour channels, and wrong
   rotations are invisible in a loss curve and obvious in a picture.
2. **Parameter count.** Catches a silent typo in the model width or depth.
3. **Overfitting one image.** 200 steps on one fixed crop with one fixed noise
   sample. The loss must fall towards zero. A model that cannot memorise a
   single image has a bug in the forward pass, the loss, or the optimiser step,
   and this finds it in two minutes instead of ten hours.
4. **Resume.** Re-runs the same command and requires `RESUMED from ... at step
   200` in the output. This is the most valuable cell in the notebook, because
   broken resume is invisible until the session you needed it.
5. **End-to-end result.** Runs eval and plots. Expect roughly 23 to 25 dB out
   from about 20 dB in after a few hundred steps. That is not a good number and
   it is not supposed to be. It proves the chain executes.

Do not continue past a failing cell. Fix it locally, push, re-run from the clone
cell.

## Phase 1 and Phase 2 implementation

`models/ffdnet.py` implements the colour baseline with noise-map conditioning
and the author's inference-weight layout. `models/blocks.py` and `models/fcsg.py`
implement soft Fourier bands, shared separable experts, spatial top-2 mixing,
SE fusion, and residual refinement. `src/fcsg_net/degrade.py` provides composite
samples and a reproducible 5,000-pair export.

`notebooks/phase1_phase2.ipynb` produced the FFDNet and Phase 2 evidence. The
notebook uses pretrained FFDNet weights for inference reproduction, records
actual GPU memory and compute, and stops on failed checks. A T4 result is not
an RTX 3050 measurement. `results/milestone_status.md` records these limits.

The experts operate after a lossless 2x pixel rearrangement and use depthwise
separable convolutions to meet the compute budget. Every expert executes;
top-2 mixing alone does not reduce computation. FFT cutoffs use cycles per
pixel, but inference tiles still change boundaries and context.

Phase 3's frequency and entropy losses, matched 200,000-step training, SSIM,
and LPIPS are complete. All four runs passed the comparison-table exit test.
See `results/milestone_status.md` for scores and the next experiments.
Phase 4 still needs ablations and routing figures.
The literature review is now in `notes/related.md`.

## Rough edges

Real inconsistencies in the repository, listed so you do not spend an afternoon
rediscovering one.

- **`configs/dncnn.toml`, `ffdnet.toml` still ask for 300,000 steps.** They are
  the Phase 1 Gaussian configs and are not used in Phase 3. The composite
  configs use 200,000 steps, set from the measured throughput in
  `results/milestone_run_20260925/throughput.json`.
