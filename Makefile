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

# Runs and features made only as a step of a pattern-rule chain are kept, never deleted as intermediates.
# (Not a bare .SECONDARY:, which makes every target secondary: a missing file listed as a
# prerequisite, such as a new tuning trial, would then not be built when its target exists.)
.PRECIOUS: $(D)/runs/B-%.pth $(D)/runs/C-%.pth $(D)/feats/B-%_test.npy $(D)/feats/C-%_test.npy

.PHONY: help env-info test notebook push-drive dataset stats sample-grid split label-subsets unlabeled-subsets views-preview views teacher-info sanity-pixels sanity-teacher student-info row-a teacher-ref teacher-cache throughput b-check tuning train-b train-b-c16 train-b-c32 train-c train-c-c16 train-c-c32 sensitivity-c check-runs svm-hparams probes diagnostics

help:
	@echo "Data dir: $(D)"
	@echo "Python:   $(PY)"
	@echo "Targets:  env-info test notebook push-drive | dataset stats sample-grid split label-subsets unlabeled-subsets views-preview views | teacher-info sanity-pixels sanity-teacher student-info row-a teacher-ref teacher-cache | throughput b-check tuning | train-b train-b-c16 train-b-c32 train-c train-c-c16 train-c-c32 sensitivity-c check-runs | svm-hparams probes diagnostics"

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

# ---------------------------------------------------------------------------
# Phase 2: throughput and training
# ---------------------------------------------------------------------------
# E14: images/s of each training stage on the fixed GPU, and the GPU-hours of each epoch budget
THROUGHPUT := $(D)/throughput.json
$(THROUGHPUT): $(VIEWS) $(TEACHER_FEATS) $(SPLIT) mo433/throughput.py mo433/training.py \
		mo433/student.py mo433/teacher.py mo433/transforms.py
	$(PY) -m mo433.throughput

throughput: $(THROUGHPUT)

# E15: one training of row B or C. The run name fixes everything (mo433/train.py): main runs
# <row>-c<c>-d<p>-s<s> use the row's tuned (lr, h); tuning trials end in -tune-lr<lr>-h<h>.
TRAIN_CODE := mo433/train.py mo433/training.py mo433/student.py mo433/transforms.py
$(D)/runs/B-%.pth: $(VIEWS) $(UNLAB_50) $(TRAIN_CODE)
	$(PY) -m mo433.train B-$*

$(D)/runs/C-%.pth: $(VIEWS) $(UNLAB_50) $(TEACHER_FEATS) $(TRAIN_CODE)
	$(PY) -m mo433.train C-$*

# Features of a trained student (then $(D)/sanity/<run>.json by the rule of step E9)
EXTRACT_CODE := mo433/extract_features.py mo433/features.py mo433/student.py mo433/transforms.py
$(D)/feats/B-%_test.npy: $(D)/runs/B-%.pth $(SPLIT) $(EXTRACT_CODE)
	$(PY) -m mo433.extract_features B-$*

$(D)/feats/C-%_test.npy: $(D)/runs/C-%.pth $(SPLIT) $(EXTRACT_CODE)
	$(PY) -m mo433.extract_features C-$*

# E16: first real training of row B (it is also one of the tuning trials of step E17) and its
# collapse check against row A
B_CHECK := B-c16-d100-s0-tune-lr1e-3-h2048
b-check: $(D)/sanity/$(B_CHECK).json $(D)/sanity/A-c16-s0.json

# E17: the 12 trials of each row (lr x h, 50 epochs, c16, seed 0) and the selection by k-NN
# 5-fold CV on the training split. Names must match mo433/tuning.py:trial_names.
TUNE_TRIALS := $(foreach row,B C,$(foreach lr,3e-4 1e-3 3e-3 1e-2 3e-2 1e-1,$(foreach h,1024 2048,\
	$(row)-c16-d100-s0-tune-lr$(lr)-h$(h))))
TUNING := $(D)/tuning.json
$(TUNING): $(TUNE_TRIALS:%=$(D)/feats/%_test.npy) $(SPLIT) mo433/tuning.py mo433/evaluate.py
	$(PY) -m mo433.tuning

tuning: $(TUNING)

# ---------------------------------------------------------------------------
# Phase 3: main trainings
# ---------------------------------------------------------------------------
# E18: the 10 main runs of row B with the selected (lr, h) (config.SELECTED): each seed at 50%
# and 100% of the unlabeled images, the 50% runs of each width first. Each run brings its
# features and its sanity / collapse check. Names must match mo433/train.py:main_runs.
main_runs = $(foreach p,50 100,$(foreach s,$(if $(filter 16,$(2)),0 1 2,0 1),$(1)-c$(2)-d$(p)-s$(s)))
B_C16 := $(call main_runs,B,16)
B_C32 := $(call main_runs,B,32)
train-b-c16: $(B_C16:%=$(D)/sanity/%.json) $(D)/sanity/A-c16-s0.json
train-b-c32: $(B_C32:%=$(D)/sanity/%.json) $(D)/sanity/A-c32-s0.json
train-b: train-b-c16 train-b-c32

# E19: the 10 main runs of row C, in the same order and with the same checks
C_C16 := $(call main_runs,C,16)
C_C32 := $(call main_runs,C,32)
train-c-c16: $(C_C16:%=$(D)/sanity/%.json) $(D)/sanity/A-c16-s0.json
train-c-c32: $(C_C32:%=$(D)/sanity/%.json) $(D)/sanity/A-c32-s0.json
train-c: train-c-c16 train-c-c32

# Sensitivity check of row C (appendix only, never used to select): the main runs of C show
# transient loss spikes at the selected lr (3e-2), so the c16/d100 runs are repeated, every seed,
# with the second-best lr of the tuning (1e-2), everything else unchanged (DECISOES 14)
SENS_C := $(foreach s,0 1 2,C-c16-d100-s$(s)-lr1e-2)
$(SENS_C:%=$(D)/runs/%.pth): $(D)/runs/C-c16-d100-s%-lr1e-2.pth: $(VIEWS) $(UNLAB_50) \
		$(TEACHER_FEATS) $(TRAIN_CODE)
	$(PY) -m mo433.train C-c16-d100-s$* --lr 1e-2 --tag lr1e-2

sensitivity-c: $(SENS_C:%=$(D)/sanity/%.json) $(SENS_C:%=$(D)/probes/%.json) \
		$(D)/sanity/A-c16-s0.json

# E20: the 20 main runs together (present, on the L4, full steps, collapse criterion); writes
# nothing, so it always runs
check-runs:
	$(PY) -m mo433.check_runs

# ---------------------------------------------------------------------------
# Phase 4: evaluation
# ---------------------------------------------------------------------------
# E21: C of the linear SVM and (C, gamma) of the RBF SVM, chosen once by CV on the training
# split of a single encoder (config.SVM_HPARAM_ENCODER), then frozen for all encoders
SVM_HPARAMS := $(D)/svm_hparams.json
$(SVM_HPARAMS): $(D)/feats/B-c16-d100-s0_test.npy $(SPLIT) mo433/svm_hparams.py mo433/evaluate.py
	$(PY) -m mo433.svm_hparams

svm-hparams: $(SVM_HPARAMS)

# E22: the three probes of every encoder at every label fraction and draw, with the frozen SVM
# hyperparameters (26 encoders: A 5, B 10, C 10, D 1)
ENCODERS := $(A_ENCODERS) $(B_C16) $(B_C32) $(call main_runs,C,16) $(call main_runs,C,32) D
$(D)/probes/%.json: $(D)/feats/%_test.npy $(LABEL_SUBSETS) $(SVM_HPARAMS) mo433/probe.py \
		mo433/evaluate.py
	$(PY) -m mo433.probe $*

probes: $(ENCODERS:%=$(D)/probes/%.json)

# E23: k-means (NMI, ARI) and the 2-D t-SNE of the test features of every encoder, with the same
# parameters for all; diag/<encoder>_proj.npy is written first, then the .json target
$(D)/diag/%.json: $(D)/feats/%_test.npy mo433/diagnostics.py mo433/evaluate.py
	$(PY) -m mo433.diagnostics $*

diagnostics: $(ENCODERS:%=$(D)/diag/%.json)
