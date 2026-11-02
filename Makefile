# MO433 pipeline. Each target is a file in $(D); make only rebuilds what is missing or older
# than its inputs.
#   make -n <target>   show what would run, without running it
#   make -B <target>   force a rebuild
# Phases and steps: ../PLANO-IMPLEMENTACAO.md

# Data root: $MO433_DATA if set (Colab: /content/drive/MyDrive/mo433-2s2026), else ../data
D  ?= $(or $(MO433_DATA),$(abspath ../data))
# Report folder (figures and tables are written straight into it)
REPORT ?= $(abspath ../relatorio)
# rclone remote whose root is the Drive folder mo433-2s2026 (root_folder_id), the same folder
# Colab sees at /content/drive/MyDrive/mo433-2s2026
DRIVE ?= gdrive:
# Use the project venv when it exists (laptop); plain python otherwise (Colab)
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

export MO433_DATA := $(D)

.PHONY: help env-info test notebook push-drive dataset stats sample-grid split label-subsets unlabeled-subsets views-preview views teacher-info sanity-pixels sanity-teacher student-info row-a teacher-ref teacher-cache

help:
	@echo "Data dir: $(D)"
	@echo "Python:   $(PY)"
	@echo "Targets:  env-info test notebook push-drive | dataset stats sample-grid split label-subsets unlabeled-subsets views-preview views | teacher-info sanity-pixels sanity-teacher student-info row-a teacher-ref teacher-cache"

# Always rerun: records the environment of the machine it runs on
env-info:
	$(PY) -m mo433.env_info

test:
	$(PY) -m pytest -q

# Upload the laptop's outputs to Drive and verify them by checksum (MD5 on both sides).
# The raw JPEG download and local-only files stay on the laptop.
DRIVE_EXCLUDE := --exclude "raw/**" --exclude "figures/**" --exclude "env_laptop.json" --exclude "*.tmp"
push-drive:
	rclone copy $(D) $(DRIVE) $(DRIVE_EXCLUDE) --progress
	rclone check $(D) $(DRIVE) $(DRIVE_EXCLUDE) --one-way

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

# E5: visual check of the training views (one image per class, K views, real E6 seeds)
VIEWS_PREVIEW := $(D)/figures/views_preview.png
$(VIEWS_PREVIEW): $(SPLIT) mo433/transforms.py mo433/views_preview.py
	$(PY) -m mo433.views_preview

views-preview: $(VIEWS_PREVIEW)

# E6: K=10 pre-generated views per training image (~2.65 GB, CPU, one process per core)
VIEWS := $(D)/views_K10.npy
$(VIEWS): $(SPLIT) mo433/transforms.py mo433/views.py
	$(PY) -m mo433.views

views: $(VIEWS)

# ---------------------------------------------------------------------------
# Phase 1: teacher (row D) and row A
# ---------------------------------------------------------------------------
# E7: load and freeze the DINO ViT-S/16 teacher; record its checks and weights hash
TEACHER_INFO := $(D)/teacher_info.json
$(TEACHER_INFO): mo433/teacher.py mo433/transforms.py
	$(PY) -m mo433.teacher

teacher-info: $(TEACHER_INFO)

# E9: sanity diagnostics on raw pixels (reference point; includes the shuffled-label control)
$(D)/sanity/pixels.json: $(SPLIT) $(LABEL_SUBSETS) mo433/sanity.py mo433/evaluate.py
	$(PY) -m mo433.sanity pixels

sanity-pixels: $(D)/sanity/pixels.json

# E10: cached features of the teacher (D) and of its random-weight control (D-random).
# One run writes <encoder>_train.npy and then <encoder>_test.npy (the target).
$(D)/feats/D_test.npy $(D)/feats/D-random_test.npy: $(D)/feats/%_test.npy: $(SPLIT) \
		mo433/extract_features.py mo433/features.py mo433/teacher.py mo433/transforms.py
	$(PY) -m mo433.extract_features $*

# Sanity check of any encoder with cached features (pixels has its own rule above)
$(D)/sanity/%.json: $(D)/feats/%_test.npy $(LABEL_SUBSETS) mo433/sanity.py mo433/evaluate.py
	$(PY) -m mo433.sanity $*

# E10 checkpoint: the teacher and its negative controls
sanity-teacher: $(D)/sanity/D.json $(D)/sanity/D-random.json $(D)/sanity/pixels.json

# E11: student family (params, FLOPs, stage shapes, head sizes) and row A (untrained students)
STUDENT_INFO := $(D)/student_info.json
$(STUDENT_INFO): mo433/student.py mo433/teacher.py
	$(PY) -m mo433.student

student-info: $(STUDENT_INFO)

A_ENCODERS := A-c16-s0 A-c16-s1 A-c16-s2 A-c32-s0 A-c32-s1
A_FEATS := $(A_ENCODERS:%=$(D)/feats/%_test.npy)
$(A_FEATS): $(D)/feats/%_test.npy: $(SPLIT) mo433/extract_features.py mo433/features.py \
		mo433/student.py mo433/transforms.py
	$(PY) -m mo433.extract_features $*

row-a: $(STUDENT_INFO) $(A_ENCODERS:%=$(D)/sanity/%.json)

# E13: teacher features of every view (fp16), the targets of row C. The reference (20 views,
# fp32) is computed on the laptop's CPU; the cache is computed on Colab's GPU and checked
# against it (and against fp32 on the GPU) before it is written.
TEACHER_REF := $(D)/teacher_ref.npz
$(TEACHER_REF): $(VIEWS) mo433/teacher_cache.py mo433/features.py mo433/teacher.py \
		mo433/transforms.py
	$(PY) -m mo433.teacher_cache --reference

teacher-ref: $(TEACHER_REF)

TEACHER_FEATS := $(D)/teacher_feats_K10.npy
$(TEACHER_FEATS): $(VIEWS) $(TEACHER_REF) $(TEACHER_INFO) mo433/teacher_cache.py \
		mo433/features.py mo433/teacher.py mo433/transforms.py
	$(PY) -m mo433.teacher_cache

teacher-cache: $(TEACHER_FEATS)

# Report figure (appendix): random images per class
$(REPORT)/figures/eurosat_samples.pdf: $(EUROSAT) mo433/sample_grid.py
	$(PY) -m mo433.sample_grid --out $@

sample-grid: $(REPORT)/figures/eurosat_samples.pdf
