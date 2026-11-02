"""Fixed settings of the study, in one place.

Every value here is a decision recorded in relatorio/DECISOES.md (the number in brackets) or
a requirement of the project specification (mo433_project.pdf, section in brackets).
Values that are only fixed after the throughput measurement (week 2) are None until then.
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

# Fixed split of Neumann et al. (2019), distributed with TorchGeo (DECISOES 6).
# train + val lists are merged into our training split; the test list is the test split.
SPLIT_URL = "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{}.txt"
SPLIT_LISTS = {"train": ("train", "val"), "test": ("test",)}
N_TRAIN, N_TEST = 21_600, 5_400

# Labeled subsets for the probes (spec 3.1)
LABEL_FRACTIONS = (0.01, 0.10, 0.50, 1.00)
LABEL_DRAWS = (0, 1, 2)

# Unlabeled subsets for training B and C; 50% is the ablation level (spec 5, DECISOES 15)
UNLAB_FRACTIONS = (100, 50)
UNLAB_SEED = 0

# ---------------------------------------------------------------------------
# Augmentation and fixed views (spec 3.1, 3.9; DECISOES 11-14, 31)
# ---------------------------------------------------------------------------
K_VIEWS = 10
VIEW_SEED = 0
MIN_SCALE = 0.25
# VICRegViewTransform defaults to sigmas=(0.2, 2), but VICRegTransform passes (0.1, 2).
BLUR_SIGMAS = (0.1, 2.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Teacher, row D (spec 3.3; DECISOES 8-10)
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
# Student family (spec 2.1; DECISOES 1-5)
# ---------------------------------------------------------------------------
WIDTHS = (16, 32)                      # c; stages [c, 2c, 4c, 8c], output dim 8c
SEEDS = {16: (0, 1, 2), 32: (0, 1)}    # spec 2.1: >=3 seeds small, >=2 seeds large

# ---------------------------------------------------------------------------
# Training of B and C (spec 3.4, 3.5; DECISOES 18-19)
# ---------------------------------------------------------------------------
# Fixed from the throughput measurement (E14, throughput.json): 200 epochs of every main run
# cost ~4.7 GPU-h of steps on the L4, so the budget is not binding (DECISOES 18).
EPOCHS = 200
BATCH_SIZE = 512                       # same throughput as 256, half the steps
WARMUP_FRACTION = 0.05                 # linear warm-up over 5% of the steps (10 of 200 epochs)
WEIGHT_DECAY = 0.05
TUNING_LRS = (3e-4, 1e-3, 3e-3)
TUNING_HIDDEN = (1024, 2048)
TUNING_EPOCHS = 50                     # a quarter of the final length (DECISOES 19)
TUNING_WIDTH, TUNING_SEED = 16, 0
# (lr, h) selected for each row by the tuning (step E17); None until then
SELECTED = {"B": None, "C": None}

# Throughput measurement (step E14; spec 3.9): candidate batch sizes, largest head of the grid
THROUGHPUT_BATCHES = (256, 512)
THROUGHPUT_HIDDEN = max(TUNING_HIDDEN)
THROUGHPUT_WARMUP, THROUGHPUT_ITERS = 5, 30
THROUGHPUT_EPOCH_OPTIONS = (50, 100, 200, 300)   # budgets tabulated from the measurement

# ---------------------------------------------------------------------------
# Evaluation (spec 3.6-3.8; DECISOES 20-28)
# ---------------------------------------------------------------------------
KNN_MAX_K = 20                         # k = min(20, n of the smallest class in the subset)
SVM_CV_FOLDS = 5
SVM_C_GRID = (0.01, 0.1, 1, 10, 100)
SVM_G_GRID = (0.1, 0.3, 1, 3, 10)      # gamma = g / d
SVM_HPARAM_ENCODER = "B-c16-d100-s0"
LINEAR_SVC_MAX_ITER = 10_000
RBF_MAX_SAMPLES = 10_000
RBF_SUBSAMPLE_SEED = 0
KMEANS_N_INIT = 10
KMEANS_SEED = 0
TSNE_PERPLEXITY = 30
TSNE_SEED = 0
LATENCY_BATCH = 256
LATENCY_WARMUP = 10
LATENCY_ITERS = 100
