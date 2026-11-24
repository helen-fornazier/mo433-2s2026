# MO433 — Distilling unsupervised representations into small models

Is distilling a frozen self-supervised ViT (DINO ViT-S/16) into a small convnet (row C) better
than training the same convnet directly with VICReg (row B)? Budget-matched comparison on
EuroSAT, with a random-init baseline (A) and the teacher itself (D).

The report and the design decisions are in `../relatorio/` (`main.tex`, `DECISOES.md`), and the
step-by-step plan is in `../PLANO-IMPLEMENTACAO.md`.

## Layout

| Path | Content |
|---|---|
| `mo433/` | one module per pipeline step, runnable as `python -m mo433.<step>` |
| `mo433/config.py` | every fixed setting of the study (seeds, K, fractions, grids…) |
| `tests/` | quick checks, run on the laptop with `make test` |
| `notebooks/` | exploration only (`make notebook`); nothing in the pipeline depends on them |
| `Makefile` | the pipeline: dependencies between steps, skips what already exists |

Data and outputs are never stored in the repository. They live in `$MO433_DATA`, which
defaults to `../data` on the laptop and is `/content/drive/MyDrive/mo433-2s2026` on Colab.
Every output has a `<name>.json` next to it with the config, git commit, package versions and
device that produced it.

## Setup

Laptop (CPU; data preparation, probes, figures):

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt jupyterlab
.venv/bin/pip install -e .     # so that `import mo433` works from any folder (notebooks)
.venv/bin/pip install nbstripout && .venv/bin/nbstripout --install   # never commit notebook outputs
make env-info test
```

Colab (GPU; teacher cache and trainings): upload `notebooks/colab_run.ipynb` to Colab
(*File → Upload notebook*), select an **L4** GPU and add a read-only `GITHUB_TOKEN` to Colab
Secrets. The notebook mounts Drive, clones this repository, installs `requirements.txt` (keeping
Colab's CUDA build of torch) and runs `make <target>` against `/content/drive/MyDrive/mo433-2s2026`.

On Colab, `make` only builds outputs that are **missing** on Drive: after cloning, the notebook
gives every code file an old fixed date, so code never looks newer than data. To redo a step there,
delete its output. Do not use `make -B` on Colab, because it rebuilds the whole chain of prerequisites.
On the laptop, where the code is edited, make's usual timestamp rule applies.

## Reproducing

`make -n <target>` shows what would run. The whole pipeline is split by where it runs:

| Target | Runs on | What it builds |
|---|---|---|
| `make prep` | laptop | EuroSAT arrays, split, labeled and unlabeled subsets, the K=10 views, teacher info and fp32 reference, student info |
| `make results-gpu` | Colab (L4) | teacher cache (row C's targets), rows A and D, the 20 main runs of B and C with their features and sanity / collapse checks |
| `make results-cpu` | laptop | SVM hyperparameters, the probes and the diagnostics of the 26 encoders |
| `make latency` | laptop, idle (~40 min) | `tab:encoders` latency |
| `make report` | laptop | every results table and figure, in `../relatorio/tables` and `../relatorio/figures` |

`make all` runs all of them in order (where each one can run). Design steps whose result is
frozen in `mo433/config.py` are separate: `make throughput` (epochs, batch size) and
`make tuning` (learning rate and projector width of each row), as is the appendix-only
`make sensitivity-c`.

The results of every step are in `$MO433_DATA`; the report is compiled from `../relatorio`
with `tectonic main.tex`.

### From-scratch check (step E27)

In a new, empty data folder, from a clean clone:

1. Laptop: `make prep D=../data-repro`, then upload it into the Drive folder
   `mo433-2s2026/repro` (`rclone copy ../data-repro gdrive:repro --exclude "raw/**"`).
2. Colab: in `notebooks/colab_run.ipynb` set `DATA = "mo433-2s2026/repro"` and run
   `make results-gpu` (or `teacher-cache row-a sanity-teacher` first, then `train-b-c16`,
   `train-b-c32`, `train-c-c16`, `train-c-c32` in parallel sessions).
3. Laptop: copy the results back without the large training inputs
   (`rclone copy gdrive:repro ../data-repro --exclude "raw/**" --exclude "views_K10.npy" --exclude "teacher_feats_K10.npy"`),
   then `make results-cpu D=../data-repro` and `make repro-compare D=../data-repro`, which
   compares every cell of `tab:main` with the original results and writes
   `../data-repro/repro_compare.json`.
