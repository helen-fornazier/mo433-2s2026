# MO433 pipeline. Each target is a file in $(D); make only rebuilds what is missing or older
# than its inputs.
#   make -n <target>   show what would run, without running it
#   make -B <target>   force a rebuild
# Phases and steps: ../PLANO-IMPLEMENTACAO.md

# Data root: $MO433_DATA if set (Colab: /content/drive/MyDrive/mo433-2s2026), else ../data
D  ?= $(or $(MO433_DATA),$(abspath ../data))
# Use the project venv when it exists (laptop); plain python otherwise (Colab)
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

export MO433_DATA := $(D)

.PHONY: help env-info test dataset

help:
	@echo "Data dir: $(D)"
	@echo "Python:   $(PY)"
	@echo "Targets:  env-info test | dataset"

# Always rerun: records the environment of the machine it runs on
env-info:
	$(PY) -m mo433.env_info

test:
	$(PY) -m pytest -q

# ---------------------------------------------------------------------------
# Phase 0: data
# ---------------------------------------------------------------------------
EUROSAT := $(D)/eurosat/images.npy

# E1: EuroSAT JPEGs -> images.npy, labels.npy, index.json; TorchGeo split lists
$(EUROSAT): mo433/dataset.py
	$(PY) -m mo433.dataset

dataset: $(EUROSAT)
