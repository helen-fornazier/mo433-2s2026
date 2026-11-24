"""The two transform pipelines of the study (spec 3.1) and the teacher's preprocessing.

Images are kept as uint8 (N, 64, 64, 3) arrays everywhere (dataset and pre-generated views)
and normalized as tensors at the last moment, on whatever device the batch is on.

* Training transform: lightly's VICReg augmentation at 64 px with min_scale 0.25
  (report Sec. 5.1), one view per call. ``generate_view`` applies it with a fixed seed per
  (image, view), so that the K pre-generated views (step E6) are reproducible.
* Evaluation transform: deterministic, tensor conversion + ImageNet normalization; images are
  already 64 px, so there is no resize or crop (report Sec. 5.1).
* Teacher preprocessing: bicubic upsampling to 224 px + ImageNet normalization (report Sec. 3.2).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from lightly.transforms import VICRegTransform, VICRegViewTransform
from PIL import Image

from mo433 import config

# Arguments of the training augmentation. Everything else stays at lightly's defaults.
# sigmas: VICRegViewTransform defaults to (0.2, 2), VICRegTransform passes (0.1, 2) (report Sec. 5.1).
VIEW_KWARGS = dict(input_size=config.IMAGE_SIZE, min_scale=config.MIN_SCALE, normalize=None)


def make_view_transform() -> VICRegViewTransform:
    """Single-view VICReg augmentation; returns float tensors in [0, 1] (no normalization)."""
    return VICRegViewTransform(**VIEW_KWARGS, sigmas=config.BLUR_SIGMAS)


def reference_view_transform() -> VICRegViewTransform:
    """The per-view transform that VICRegTransform itself builds with the same arguments."""
    return VICRegTransform(**VIEW_KWARGS).transforms[0]


def describe(transform) -> list:
    """Flat, comparable description of a (nested) torchvision/lightly transform: class names
    and their scalar or tuple attributes. Used to check that two pipelines are identical."""
    if isinstance(transform, VICRegViewTransform):
        return describe(transform.transform)
    attrs = {k: v for k, v in sorted(vars(transform).items())
             if not k.startswith("_") and k != "training"
             and isinstance(v, (int, float, str, bool, tuple, list)) and k != "transforms"}
    out = [(type(transform).__name__, attrs)]
    for child in getattr(transform, "transforms", None) or []:   # Compose, RandomApply
        out += describe(child)
    return out


def view_seed(image_index: int, view_index: int, base_seed: int = config.VIEW_SEED) -> int:
    """Independent 32-bit seed for view ``view_index`` of training image ``image_index``."""
    return int(np.random.SeedSequence([base_seed, image_index, view_index]).generate_state(1)[0])


def generate_view(image: np.ndarray, seed: int, transform=None) -> np.ndarray:
    """One augmented view of a (64, 64, 3) uint8 image, as (64, 64, 3) uint8.

    lightly draws from torch's RNG (crop, flip, jitter, grayscale) and from numpy's global RNG
    (blur, solarization); both are seeded, plus Python's ``random`` for safety. Every operation
    before ToTensor works on 8-bit PIL images, so converting back to uint8 is exact.
    """
    transform = transform or make_view_transform()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    view = transform(Image.fromarray(image))                       # float, (3, 64, 64), k/255
    return (view * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def to_tensor(images) -> torch.Tensor:
    """(N, H, W, 3) uint8 array or tensor -> (N, 3, H, W) uint8 tensor.

    Values are not copied, except for read-only arrays (e.g. a memmap opened with mode "r"),
    which torch cannot wrap safely."""
    if isinstance(images, np.ndarray) and not images.flags.writeable:
        images = images.copy()
    x = torch.as_tensor(images)
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) uint8, got {tuple(x.shape)} {x.dtype}")
    return x.permute(0, 3, 1, 2)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) uint8 tensor -> float32, ImageNet-normalized, on the same device."""
    mean = torch.tensor(config.IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255.0 - mean) / std


def eval_transform(images) -> torch.Tensor:
    """Deterministic evaluation transform for the student (and the input of the teacher's)."""
    return normalize(to_tensor(images))


def teacher_preprocess(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, 64, 64) uint8 tensor -> (N, 3, 224, 224) float32 for the teacher: bicubic
    upsampling, then ImageNet normalization. The same view as the student's, only resized."""
    up = F.interpolate(x.float() / 255.0, size=config.TEACHER_RES, mode=config.TEACHER_UPSAMPLE,
                       align_corners=False)
    mean = torch.tensor(config.IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (up - mean) / std
