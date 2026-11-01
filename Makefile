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

.PHONY: help env-info test notebook dataset stats sample-grid

help:
	@echo "Data dir: $(D)"
	@echo "Python:   $(PY)"
	@echo "Targets:  env-info test notebook | dataset stats sample-grid"

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

# Report figure (appendix): random images per class
$(REPORT)/figures/eurosat_samples.pdf: $(EUROSAT) mo433/sample_grid.py
	$(PY) -m mo433.sample_grid --out $@

sample-grid: $(REPORT)/figures/eurosat_samples.pdf
