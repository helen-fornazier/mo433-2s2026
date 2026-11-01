import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from mo433 import config
from mo433.transforms import (VIEW_KWARGS, describe, eval_transform, generate_view,
                              make_view_transform, normalize, reference_view_transform,
                              teacher_preprocess, to_tensor, view_seed)
from lightly.transforms import VICRegViewTransform

RNG = np.random.default_rng(0)
IMAGES = RNG.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)


def test_view_transform_matches_vicreg_transform():
    assert describe(make_view_transform()) == describe(reference_view_transform())
    # the check has teeth: the single-view transform's own defaults differ (sigmas)
    assert describe(VICRegViewTransform(**VIEW_KWARGS)) != describe(reference_view_transform())


def test_generate_view_is_seeded():
    a = generate_view(IMAGES[0], seed=view_seed(0, 0))
    assert a.shape == (64, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, generate_view(IMAGES[0], seed=view_seed(0, 0)))
    assert not np.array_equal(a, generate_view(IMAGES[0], seed=view_seed(0, 1)))
    assert view_seed(0, 1) != view_seed(1, 0)


def test_view_values_are_exact_uint8():
    torch.manual_seed(1); np.random.seed(1)
    view = make_view_transform()(Image.fromarray(IMAGES[1])) * 255
    assert torch.allclose(view, view.round(), atol=1e-3)


def test_eval_transform_is_deterministic_and_matches_torchvision():
    a, b = eval_transform(IMAGES), eval_transform(IMAGES)
    assert torch.equal(a, b) and a.shape == (4, 3, 64, 64)
    ref = T.Compose([T.ToTensor(), T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])
    expected = torch.stack([ref(Image.fromarray(img)) for img in IMAGES])
    assert torch.allclose(a, expected, atol=1e-6)


def test_teacher_preprocess_shape_and_normalization():
    x = to_tensor(IMAGES)
    up = teacher_preprocess(x)
    assert up.shape == (4, 3, config.TEACHER_RES, config.TEACHER_RES)
    # a constant image stays constant after bicubic upsampling, so it must equal normalize()
    const = to_tensor(np.full((1, 64, 64, 3), 100, dtype=np.uint8))
    assert torch.allclose(teacher_preprocess(const), normalize(const).mean((2, 3), keepdim=True)
                          .expand(1, 3, config.TEACHER_RES, config.TEACHER_RES), atol=1e-5)
