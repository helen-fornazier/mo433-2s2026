import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

import mo433.transforms
from mo433.features import extract
from mo433.teacher import load_teacher

IMAGES = np.random.default_rng(0).integers(0, 256, (10, 64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def teacher():
    return load_teacher(pretrained=False)


def test_batch_size_does_not_change_teacher_features(teacher):
    one = extract(teacher, IMAGES, batch_size=1)
    many = extract(teacher, IMAGES, batch_size=8)       # 8 + 2: also tests the last partial batch
    assert one.shape == (10, 384) and one.dtype == np.float32
    np.testing.assert_allclose(one, many, rtol=1e-4, atol=1e-4)


class BatchNormNet(nn.Module):
    """Tiny uint8-input model with BatchNorm, to check that extract() uses eval mode."""
    def __init__(self):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x.float() / 255)).mean((2, 3))


def test_extract_uses_eval_mode_and_order():
    net = BatchNormNet().train()
    a = extract(net, IMAGES, batch_size=3)
    assert not net.training
    # In eval mode each image's feature does not depend on the rest of the batch.
    b = extract(net, IMAGES[::-1], batch_size=4)[::-1]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_extract_cannot_augment(monkeypatch):
    assert "transform" not in inspect.signature(extract).parameters

    def forbidden(*args, **kwargs):
        raise AssertionError("augmentation called during extraction")
    monkeypatch.setattr(mo433.transforms, "generate_view", forbidden)
    monkeypatch.setattr(mo433.transforms, "make_view_transform", forbidden)
    extract(BatchNormNet(), IMAGES, batch_size=5)


def test_extract_accepts_memmap(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, IMAGES)
    net = BatchNormNet()
    np.testing.assert_array_equal(extract(net, np.load(path, mmap_mode="r")), extract(net, IMAGES))
