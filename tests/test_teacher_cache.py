import json

import numpy as np
import pytest
import torch

import mo433.teacher_cache as tc
from mo433 import config
from mo433.teacher import load_teacher, weights_sha256


def test_sample_views_are_distinct_and_fixed():
    a = tc.sample_views(100, 10, 50, seed=0)
    assert a.shape == (50, 2) and len({tuple(p) for p in a}) == 50
    assert a[:, 0].max() < 100 and a[:, 1].max() < 10
    np.testing.assert_array_equal(a, tc.sample_views(100, 10, 50, seed=0))


@pytest.fixture
def tiny_root(tmp_path, monkeypatch):
    """A data dir with 4 images x 3 views and a random-weight teacher in place of DINO."""
    def random_teacher(device="cpu", amp=True):
        torch.manual_seed(0)
        return load_teacher(device, pretrained=False, amp=amp)
    monkeypatch.setattr(tc, "load_teacher", random_teacher)
    monkeypatch.setattr(config, "TEACHER_REF_VIEWS", 4)
    monkeypatch.setattr(config, "TEACHER_FP32_VIEWS", 5)
    monkeypatch.setattr(config, "K_VIEWS", 3)
    views = np.random.default_rng(0).integers(0, 256, (4, 3, 64, 64, 3), dtype=np.uint8)
    np.save(tmp_path / "views_K3.npy", views)
    (tmp_path / "teacher_info.json").write_text(
        json.dumps({"weights_sha256": weights_sha256(random_teacher().vit)}))
    return tmp_path, views


def test_cache_end_to_end(tiny_root):
    root, views = tiny_root
    tc.write_reference(root, views)
    tc.write_cache(root, views, "cpu", batch_size=5)
    cache = np.load(root / "teacher_feats_K3.npy")
    assert cache.shape == (4, 3, config.TEACHER_DIM) and cache.dtype == np.float16
    meta = json.loads((root / "teacher_feats_K3.npy.json").read_text())
    assert meta["checks"]["cache_vs_laptop_min_cos"] > 0.999
    assert meta["checks"]["next_view_vs_laptop_max_cos"] < 0.999


def test_misaligned_cache_is_rejected(tiny_root):
    root, views = tiny_root
    tc.write_reference(root, views)
    cache = tc.extract(tc.load_teacher(), views.reshape(12, 64, 64, 3)).reshape(4, 3, -1)
    shifted = np.roll(cache, 1, axis=1).astype(np.float16)       # feature k belongs to view k-1
    with pytest.raises(SystemExit, match="cache_vs_laptop"):
        tc.check_cache(shifted, views, "cpu", root)
