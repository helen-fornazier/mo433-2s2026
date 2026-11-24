"""Fixed settings of the study, in one place.

Every value here is either a requirement of the project specification ("spec", with its section
number) or a choice justified in the report ("report Sec."), as noted next to it.
"""

# ---------------------------------------------------------------------------
# Dataset and splits
# ---------------------------------------------------------------------------
DATASET = "EuroSAT (RGB), torchvision release"
NUM_CLASSES = 10
IMAGE_SIZE = 64                        # native resolution, also the student's input size
N_IMAGES = 27_000
# Images per class in the RGB release (Helber et al., 2019); checked when the arrays are built.
EXPECTED_CLASS_COUNTS = {
    "AnnualCrop": 3000, "Forest": 3000, "HerbaceousVegetation": 3000, "Highway": 2500,
    "Industrial": 2500, "Pasture": 2000, "PermanentCrop": 2500, "Residential": 3000,
    "River": 2500, "SeaLake": 3000,
}

# Fixed split of Neumann et al. (2019), distributed with TorchGeo (report Sec. 5.1).
# train + val lists are merged into our training split; the test list is the test split.
SPLIT_URL = "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{}.txt"
SPLIT_LISTS = {"train": ("train", "val"), "test": ("test",)}
N_TRAIN, N_TEST = 21_600, 5_400

# Labeled subsets for the probes (spec 3.1)
LABEL_FRACTIONS = (0.01, 0.10, 0.50, 1.00)
LABEL_DRAWS = (0, 1, 2)

# Unlabeled subsets for training B and C; 50% is the ablation level (spec 5, report Sec. 6.5)
UNLAB_FRACTIONS = (100, 50)
UNLAB_SEED = 0

# ---------------------------------------------------------------------------
# Augmentation and fixed views (spec 3.1, 3.9; report Sec. 5.1)
# ---------------------------------------------------------------------------
K_VIEWS = 10
VIEW_SEED = 0
MIN_SCALE = 0.25
# VICRegViewTransform defaults to sigmas=(0.2, 2), but VICRegTransform passes (0.1, 2).
BLUR_SIGMAS = (0.1, 2.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Teacher, row D (spec 3.3; report Sec. 3.2)
# ---------------------------------------------------------------------------
TEACHER_NAME = "vit_small_patch16_224.dino"
TEACHER_RES = 224
TEACHER_DIM = 384
TEACHER_UPSAMPLE = "bicubic"
# Teacher cache (step E13): fp16 targets for row C, checked against fp32 recomputations
TEACHER_CACHE_BATCH = 512
TEACHER_REF_VIEWS, TEACHER_REF_SEED = 20, 0        # fp32 on the laptop's CPU, compared on Colab
TEACHER_FP32_VIEWS, TEACHER_FP32_SEED = 200, 1     # fp32 recomputed on the GPU
TEACHER_CACHE_MIN_COS = 0.999                      # fp16 cache vs fp32
TEACHER_HOSTS_MIN_COS = 0.9999                     # fp32 on Colab's GPU vs fp32 on the laptop

# ---------------------------------------------------------------------------
# Student family (spec 2.1; report Sec. 3.3)
# ---------------------------------------------------------------------------
WIDTHS = (16, 32)                      # c; stages [c, 2c, 4c, 8c], output dim 8c
SEEDS = {16: (0, 1, 2), 32: (0, 1)}    # spec 2.1: >=3 seeds small, >=2 seeds large

# ---------------------------------------------------------------------------
# Training of B and C (spec 3.4, 3.5; report Secs. 5.2, 5.3)
# ---------------------------------------------------------------------------
# Fixed from the throughput measurement (E14, throughput.json): 200 epochs of every main run
# cost ~4.7 GPU-h of steps on the L4, so the budget is not binding (report Sec. 5.3).
EPOCHS = 200
BATCH_SIZE = 512                       # same throughput as 256, half the steps
WARMUP_FRACTION = 0.05                 # linear warm-up over 5% of the steps (10 of 200 epochs)
WEIGHT_DECAY = 0.05
# Extended while the best lr of either row sits at the top of the grid, for both rows alike:
# 1e-2 after both picked 3e-3, 3e-2 after both picked 1e-2, 1e-1 after C picked 3e-2 (report Sec. 5.2)
TUNING_LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
TUNING_HIDDEN = (1024, 2048)
TUNING_EPOCHS = 50                     # a quarter of the final length (report Sec. 5.2)
TUNING_WIDTH, TUNING_SEED = 16, 0
TUNING_CV_FOLDS, TUNING_CV_SEED = 5, 0  # selection: k-NN accuracy by stratified CV on train
# (lr, h) selected for each row by the tuning (step E17, data/tuning.json): both inside the grid
SELECTED = {"B": (1e-2, 1024), "C": (3e-2, 2048)}

# Throughput measurement (step E14; spec 3.9): candidate batch sizes, largest head of the grid
THROUGHPUT_BATCHES = (256, 512)
THROUGHPUT_HIDDEN = max(TUNING_HIDDEN)
THROUGHPUT_WARMUP, THROUGHPUT_ITERS = 5, 30
THROUGHPUT_EPOCH_OPTIONS = (50, 100, 200, 300)   # budgets tabulated from the measurement

# ---------------------------------------------------------------------------
# Evaluation (spec 3.6-3.8; report Sec. 4)
# ---------------------------------------------------------------------------
KNN_MAX_K = 20                         # k = min(20, n of the smallest class in the subset)
SVM_CV_FOLDS = 5
SVM_C_GRID = (0.01, 0.1, 1, 10, 100)
SVM_G_GRID = (0.1, 0.3, 1, 3, 10)      # gamma = g / d
SVM_HPARAM_ENCODER = "B-c16-d100-s0"
# Chosen by the step E21 (data/svm_hparams.json), both inside the grid, then frozen for every
# encoder: the linear SVM's CV curve is flat (93.3-93.5%); RBF g = 1 is gamma="scale"
SVM_SELECTED = {"linear": {"C": 0.1}, "rbf": {"C": 1, "g": 1}}
LINEAR_SVC_MAX_ITER = 10_000
RBF_MAX_SAMPLES = 10_000
RBF_SUBSAMPLE_SEED = 0
KMEANS_N_INIT = 10
KMEANS_SEED = 0
TSNE_PERPLEXITY = 30
TSNE_SEED = 0
# Report (step E25)
COST_WIDTH, COST_FRACTION = 16, 0.01   # tab:cost: linear SVM, small student, 1% labels
CONFUSION_FRACTION = 0.01              # fig:confusion: linear SVM, draw 0
# DINO ViT-S/16 pretraining [39]: the paper gives 72.6 h on 16 GPUs for 300 epochs (Tab. 8,
# multi-crop 2x224 + 10x96; GPU model not stated); the public checkpoint has 800 epochs, and
# the linear extrapolation to 800 epochs is ours (report, sec:teacher)
TEACHER_PRETRAIN_GPU_HOURS_300 = 72.6 * 16
TEACHER_PRETRAIN_EPOCHS = 800
TEACHER_PRETRAIN_GPU_HOURS = TEACHER_PRETRAIN_GPU_HOURS_300 * TEACHER_PRETRAIN_EPOCHS / 300
LATENCY_BATCH = 256
LATENCY_WARMUP = 10
LATENCY_ITERS = 100
LATENCY_ROUNDS = 2                     # the whole measurement twice in a row
LATENCY_MAX_ROUND_DIFF = 0.05          # rounds farther apart than this: measure again
