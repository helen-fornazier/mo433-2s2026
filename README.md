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
.venv/bin/pip install -r requirements.txt
make env-info test
```

Colab (GPU; teacher cache and trainings): see `notebooks/colab_run.ipynb` (step E12).

## Reproducing

`make -n <target>` shows what would run, and `make -B <target>` forces a rebuild. The targets
for each table and figure will be listed here as the steps are implemented.
