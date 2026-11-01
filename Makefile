# MO433 pipeline. Each target is a file in $(D); make only rebuilds what is missing or older
# than its inputs.
#   make -n <target>   show what would run, without running it
#   make -B <target>   force a rebuild
# Phases and steps: ../PLANO-IMPLEMENTACAO.md

# Data root: $MO433_DATA if set (Colab: /content/drive/MyDrive/mo433-2s2026), else ../data
D  ?= $(or $(MO433_DATA),$(abspath ../data))
# Report folder (figures and tables are written straight into it)
REPORT ?= $(abspath ../relatorio)
# Use the project venv when it exists (laptop); plain python otherwise (Colab)
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

export MO433_DATA := $(D)

.PHONY: help env-info test notebook dataset stats sample-grid split label-subsets unlabeled-subsets

help:
	@echo "Data dir: $(D)"
	@echo "Python:   $(PY)"
	@echo "Targets:  env-info test notebook | dataset stats sample-grid split label-subsets unlabeled-subsets"

# Always rerun: records the environment of the machine it runs on
env-info:
	$(PY) -m mo433.env_info

test:
	$(PY) -m pytest -q

# Exploration notebooks (not part of the pipeline)
notebook:
	$(PY) -m jupyter lab notebooks/

# ---------------------------------------------------------------------------
# Phase 0: data
# ---------------------------------------------------------------------------
EUROSAT := $(D)/eurosat/images.npy

# E1: EuroSAT JPEGs -> images.npy, labels.npy, index.json; TorchGeo split lists
$(EUROSAT): mo433/dataset.py
	$(PY) -m mo433.dataset

dataset: $(EUROSAT)

# E1b: descriptive statistics for the report (the notebook adds the figures)
$(D)/eurosat/stats.json: $(EUROSAT) mo433/data_stats.py
	$(PY) -m mo433.data_stats

stats: $(D)/eurosat/stats.json

# E2: fixed train/test split (TorchGeo train+val -> train, test -> test)
SPLIT := $(D)/split_idx.pkl
$(SPLIT): $(EUROSAT) mo433/split.py
	$(PY) -m mo433.split

split: $(SPLIT)

# E3: labeled subsets for the probes (fractions x draws), indices into the training split
LABEL_SUBSETS := $(D)/label_subsets.pkl
$(LABEL_SUBSETS): $(SPLIT) mo433/label_subsets.py
	$(PY) -m mo433.label_subsets

label-subsets: $(LABEL_SUBSETS)

# E4: unlabeled subsets (100% and the 50% ablation level), drawn without labels.
# Both files are written by one run; unlab_idx_50.pkl is written last and is the target.
UNLAB_50 := $(D)/unlab_idx_50.pkl
$(UNLAB_50): $(SPLIT) mo433/unlabeled_subsets.py
	$(PY) -m mo433.unlabeled_subsets

unlabeled-subsets: $(UNLAB_50)

# Report figure (appendix): random images per class
$(REPORT)/figures/eurosat_samples.pdf: $(EUROSAT) mo433/sample_grid.py
	$(PY) -m mo433.sample_grid --out $@

sample-grid: $(REPORT)/figures/eurosat_samples.pdf
