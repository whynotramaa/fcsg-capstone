# FCSG-Net

Band-adaptive sparse expert routing with interpretable gates, under 5M
parameters. B.Tech research project, Autumn 2026-27. Read `plan.md` for the
phase plan, the exit test on each phase, and the reading list.

## Finish the Phase 1 and Phase 2 checks

Import [notebooks/phase1_phase2.ipynb](notebooks/phase1_phase2.ipynb) into Kaggle.
Enable a GPU and Internet, attach `joe1995/div2k-dataset`, then run all cells.
The notebook bundles the current source, so it does not need a Git push or
repository credentials. It downloads the author's FFDNet weights and CBSD68
images, checks the published benchmark, runs the architecture checks, and
exports 5,000 paired crops with degradation metadata.

Download the result and snapshot archives linked by its final cell. The
notebook stops on a failed check. Read
[results/milestone_status.md](results/milestone_status.md) for the exact gates
and the distinction between implementation and measured completion.

After editing source files, rebuild the bundled notebook with
`python notebooks/build_milestones.py`. This command only packages text;
model execution remains on Kaggle or Colab.

## Where the code runs

Model code runs on Kaggle or Colab. Edit and package source locally, then use
the bundled milestone notebook or push changes for the existing training notebooks.

| Machine | Job |
|---------|-----|
| Local | Edit, inspect syntax, and package source. No model execution. |
| Kaggle | Gate checks and all long training runs. 30 GPU-hours per week, 12-hour session cap. |
| Colab | Overflow, reserved for the Phase 4 ablations. |

The existing training notebooks clone this repository over HTTPS, so they need no credentials.
When you change code, push it and re-run the clone cell. Do not edit code inside
a notebook: the next session starts from a fresh container and the edits are
gone.

## Run the gate checks

Run `notebooks/checks.ipynb` before any training run. It takes about two
minutes of GPU time and it verifies four things: the crops look right, the model
is the size it should be, the model can memorise a single image, and a killed
run resumes at the right step.

1. Open `notebooks/checks.ipynb` on Kaggle.
2. Add a public DIV2K **dataset** as an input (Add Input > Datasets, not
   Notebooks), and turn on the GPU accelerator. The dataset must contain both
   splits, e.g. `DIV2K_train_HR` and `DIV2K_valid_HR`.
3. Run every cell in order.

If a cell fails, fix the code locally, push, then re-run from the clone cell. Do
not continue past a failing cell.

## Train

Run `notebooks/kaggle_train.ipynb`. Checkpoints, the CSV log, and the figures
land in `/kaggle/working`, which persists as the notebook output.

Kaggle kills a session at 12 hours, so a full run takes three or four sessions.
To continue, attach the previous session's output as an input dataset and run
the resume cell. `training/train.py` always resumes from the highest-numbered
checkpoint it finds in `--out`.

Download `/kaggle/working/results` before you close a session, then commit the
CSV and the figures.

## Command reference

Every script takes its paths as flags, so the same commands work on Kaggle, on
Colab, and anywhere else.

```
python src/fcsg_net/data.py --data <train_HR dir> --out crops.png
python models/dncnn.py
python training/train.py --config configs/dncnn.toml --data /kaggle/input --out <ckpt dir>
python evaluation/eval.py --ckpt <ckpt.pt> --data /kaggle/input --out results
python evaluation/plots.py --csv <ckpt dir>/train_log.csv --out results/figures
```

`--data` accepts any directory above the DIV2K images.
`fcsg_net.utils.resolve_div2k` walks down from there and picks the training and
validation directories by name, so no Kaggle dataset slug is written down
anywhere. When it fails, it lists every directory that holds images. Pass
`--train-dir` and `--val-dir` with two of those paths to skip the search.

Two flags on `train.py` exist for the gate checks. `--steps N` caps the
iteration count. `--overfit-one-image` trains on one fixed crop with one fixed
noise sample, which drives the loss towards zero unless the model has a bug.

## Layout

```
src/fcsg_net/   data.py  degrade.py  metrics.py  utils.py
models/         dncnn.py  ffdnet.py  fcsg.py  blocks.py
training/       train.py         config-driven, one entry point for every model
evaluation/     eval.py  plots.py  checks.py  run_milestones.py
configs/        dncnn.toml  ffdnet.toml  fcsg.toml
notebooks/      checks.ipynb  kaggle_train.ipynb  colab_train.ipynb
                phase1_phase2.ipynb  build_milestones.py
results/        benchmark.csv  figures/
```

Write for Python 3.11. That is what Kaggle and Colab run, whatever
`.python-version` says.
