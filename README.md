# MO433 — Distilling unsupervised representations into small models

Final project of MO433 (Unsupervised Learning, UNICAMP, 2nd semester 2026).

Is distilling a frozen self-supervised ViT (DINO ViT-S/16) into a small convnet (row C) better
than training the same convnet directly with VICReg (row B)? Budget-matched comparison on
EuroSAT, with a random-init baseline (row A) and the teacher itself (row D), for two student
widths (c16, 0.70M parameters; c32, 2.80M). Every design choice and every result is described in
the report, submitted with this repository; comments in the code point to its sections as
"report Sec. X" and to the sections of the project specification as "spec X".

## Layout

| Path | Content |
|---|---|
| `mo433/` | one module per pipeline step, runnable as `python -m mo433.<module>` |
| `mo433/config.py` | every fixed setting of the study (seeds, K, fractions, grids, selected hyperparameters…) |
| `Makefile` | the pipeline: dependencies between steps; skips what already exists |
| `tests/` | quick checks of every module, run on a CPU with `make test` |
| `notebooks/colab_run.ipynb` | runs a `make` target on a Colab GPU |
| `notebooks/explore_eurosat.ipynb` | data exploration only; nothing in the pipeline depends on it |

Data and outputs are never stored in the repository. They live in `$MO433_DATA` (the `D=`
variable of the Makefile), which defaults to `../data`. Every output has a `<name>.json` next to
it with the configuration, git commit, package versions and device that produced it. The
report's tables and figures are written to `$REPORT/tables` and `$REPORT/figures`
(default `../relatorio`, the folder of the report's LaTeX source).

`results/` is a read-only snapshot (1.8 MB) of the small outputs behind the report, copied from
`$MO433_DATA` with their metadata files: probes, diagnostics, sanity checks, training logs
(including the tuning trials), SVM hyperparameter search, tuning, throughput, latency,
supplementary analyses, teacher and student info, labeled-subset summary and dataset
statistics. Every number of the report can be checked there without running anything; the
pipeline never reads it.

Naming used throughout: an encoder instance is `<row>-c<width>-d<percent>-s<seed>`, e.g.
`B-c16-d100-s0` is row B, width c=16, trained on 100% of the unlabeled training images, seed 0;
untrained row-A students have no `d` tag (`A-c16-s0`); `D` is the teacher and `D-random` the
same ViT with random weights (a negative control). The label fractions of the probes are 1%,
10%, 50% and 100%, with three labeled subsets ("draws") per fraction below 100%.

## Setup

CPU machine (data preparation, probes, diagnostics, latency, figures), Python ≥ 3.11:

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
make env-info test
```

GPU (teacher cache and trainings): any CUDA machine with the same setup (and a CUDA build of
torch), or Colab: upload `notebooks/colab_run.ipynb` (*File → Upload notebook*), select an L4
GPU, set `DATA` to a folder of your Google Drive that holds the outputs of `make prep`, and run
the cells. The notebook clones this repository, installs `requirements.txt` (keeping Colab's
CUDA build of torch) and runs `make <TARGET>`. There, the code files are given an old fixed
date after cloning, so `make` only builds outputs that are **missing**; to redo a step, delete
its output (never `make -B`, which rebuilds the whole chain of prerequisites).

## Reproducing every table and figure

`make -n <target>` shows what would run. The pipeline is split by where it runs:

| Target | Runs on | What it builds |
|---|---|---|
| `make prep` | CPU | EuroSAT arrays and statistics, train/test split, labeled and unlabeled subsets, the K=10 views per training image, teacher checks and fp32 reference, student family |
| `make results-gpu` | GPU | teacher cache (row C's targets), rows A and D, the 20 main trainings of B and C, with their features and sanity / collapse checks |
| `make results-cpu` | CPU | SVM hyperparameters, the three probes and the diagnostics of the 26 encoders |
| `make supplementary` | CPU | the paired C-vs-D tests, the final distillation cosine and the loss-spike statistics of the discussion |
| `make latency` | CPU, idle machine (~40 min) | the latency column of the encoders table |
| `make report` | CPU | every results table and figure of the report |

`make all` runs them in order on one machine (Linux; the latency pins CPU cores). To split the work between a CPU machine and a GPU,
run `make prep` on the CPU, copy `$MO433_DATA` to the GPU machine (except `raw/`), run
`make results-gpu` there, copy the results back and run the remaining targets.

Design steps whose result is frozen in `mo433/config.py` are not part of `make all`, because
their outcome had to be fixed before the main runs: `make throughput` (number of epochs and batch
size, report Sec. 5.3) and `make tuning` (learning rate and projector width of each row, report
Sec. 5.2). The sensitivity check of row C reported in the limitations is `make sensitivity-c` (report
Sec. 8). The selected values are copied into `config.py` by hand; `probe` and `train` stop if
they disagree with `svm_hparams.json` or `tuning.json`. `config.py` is deliberately not a
Makefile prerequisite: it holds frozen decisions, and changing one means deleting the outputs
it affects.

"laptop" and "Colab" in the code and comments stand for the CPU machine and the GPU machine.

### Where each table and figure comes from

Generated by `make report` (`mo433/report.py`): the encoders, sanity, main results, gap,
diagnostics, ablation and cost tables, the three appendix tables, and every results figure.
`make sample-grid` draws the EuroSAT samples. The setup tables typed in the report come from:

| Report | Source (in `$MO433_DATA`) | Target |
|---|---|---|
| labeled subsets (sizes, k of k-NN) | `label_subsets.pkl.json` | `label-subsets` |
| dataset statistics, class balance, duplicates | `eurosat/stats.json` | `stats` |
| training configuration | `mo433/config.py`, `runs/<run>_log.json` | `train-b`, `train-c` |
| tuning grid and selection | `tuning.json` | `tuning` |
| throughput | `throughput.json` | `throughput` |
| SVM hyperparameters (CV accuracies) | `svm_hparams.json` | `svm-hparams` |
| C vs D tests, cosine, loss spikes (Secs. 7–8) | `supplementary.json` | `supplementary` |
| sensitivity check of row C (Sec. 8) | `probes/C-c16-d100-s*-lr1e-2.json` | `sensitivity-c` |

**Determinism.** The data preparation is bit-for-bit reproducible (it was regenerated from a
clean clone and compared). The trainings are seeded, but cuDNN kernels are not bitwise
deterministic, so a rerun with the same seed differs slightly. `make repro-compare D=<rerun>
REF=<original>` compares every cell of the main results table of a rerun with the original,
relative to the original seed-to-seed spread, and writes `<rerun>/repro_compare.json`.

## Pipeline steps

The module docstrings, the Makefile and the printed banners number the steps E1–E27, in the
order in which they were built. E12 (the Colab runner) and E26 (writing the report) have no
module of their own.

| Step | Module | `make` target | Output in `$MO433_DATA` | Report |
|---|---|---|---|---|
| E1 | `dataset` | `dataset` | `eurosat/` (image and label arrays), `splits/` (file lists) | Sec. 5.1 |
| E1b | `data_stats` | `stats` | `eurosat/stats.json` | Sec. 5.1 |
| E2 | `split` | `split` | `split_idx.pkl` | Sec. 5.1 |
| E3 | `label_subsets` | `label-subsets` | `label_subsets.pkl` | Sec. 4, Tab. "Labeled subsets" |
| E4 | `unlabeled_subsets` | `unlabeled-subsets` | `unlab_idx_100.pkl`, `unlab_idx_50.pkl` | Sec. 6.5 |
| E5 | `views_preview` | `views-preview` | `figures/views_preview.png` | — |
| E6 | `views` | `views` | `views_K10.npy` | Sec. 5.1 |
| E7 | `teacher` | `teacher-info` | `teacher_info.json` | Sec. 3.2 |
| E8 | `features`, `extract_features` | (used by E10, E11, E15) | `feats/<encoder>_{train,test}.npy` | Sec. 4 |
| E9–E10 | `sanity`, `evaluate` | `sanity-pixels`, `sanity-teacher` | `sanity/<encoder>.json` | Sec. 6.1 |
| E11 | `student` | `student-info`, `row-a` | `student_info.json`, row A features | Secs. 3.3–3.4 |
| E13 | `teacher_cache` | `teacher-ref`, `teacher-cache` | `teacher_ref.npz`, `teacher_feats_K10.npy` | Sec. 5.1 |
| E14 | `throughput` | `throughput` | `throughput.json` | Sec. 5.3 |
| E15 | `train`, `training` | (one run: `runs/<run>.pth`) | `runs/<run>.pth`, `runs/<run>_log.json` | Secs. 3.5–3.6, 5.2 |
| E16 | — | `b-check` | first row-B run and its collapse check | — |
| E17 | `tuning` | `tuning` | `tuning.json` (24 trials) | Sec. 5.2 |
| E18–E19 | `train` | `train-b`, `train-c` (or per width: `train-b-c16`, …) | the 20 main runs | Sec. 6 |
| E20 | `check_runs` | `check-runs` | (prints) all main runs present, complete, not collapsed | Sec. 6.1 |
| E21 | `svm_hparams` | `svm-hparams` | `svm_hparams.json` | Sec. 4 |
| E22 | `probe` | `probes` | `probes/<encoder>.json` | Secs. 6.2–6.3 |
| E23 | `diagnostics` | `diagnostics` | `diag/<encoder>.json`, `diag/<encoder>_proj.npy` | Sec. 6.4 |
| E24 | `latency` | `latency` | `latency.json` | Sec. 4 |
| — | `supplementary` | `supplementary` | `supplementary.json` | Secs. 7–8 |
| E25 | `report` | `report` | `$REPORT/tables/*.tex`, `$REPORT/figures/*.pdf` | all |
| E27 | `repro_compare` | `repro-compare` | `repro_compare.json` | Sec. 5.5 |

`make help` lists every target.
