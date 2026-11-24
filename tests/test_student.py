import numpy as np
import pytest
import torch
import torch.nn as nn

from mo433 import config
from mo433.features import extract
from mo433.student import (BasicBlock, count_params, create_projector, create_student,
                           create_vicreg_head, stage_shapes)


@pytest.mark.parametrize("c, params, dim", [(16, 0.700e6, 128), (32, 2.795e6, 256)])
def test_size_family(c, params, dim):
    s = create_student(c, seed=0)
    assert abs(count_params(s) - params) / params < 0.005
    assert s.feature_dim == dim
    sizes = [shape[1] for shape in stage_shapes(s)]
    assert sizes == [64, 64, 32, 16, 8]


def test_same_initialization_in_rows_a_b_c():
    """A-s, B-s and C-s start from identical backbone weights (report Sec. 3.4)."""
    for c in config.WIDTHS:
        a = create_student(c, seed=1)
        b = create_student(c, seed=1); create_vicreg_head(c, 1024)
        cc = create_student(c, seed=1); create_projector(c, 1024)
        for other in (b, cc):
            sa, so = a.state_dict(), other.state_dict()
            assert sa.keys() == so.keys()
            assert all(torch.equal(sa[k], so[k]) for k in sa)
    w0 = create_student(16, seed=0).backbone.stem[0].weight
    w1 = create_student(16, seed=1).backbone.stem[0].weight
    assert not torch.equal(w0, w1)


def test_initialization_rules():
    s = create_student(16, seed=0)
    for m in s.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert torch.all(m.weight == 1) and torch.all(m.bias == 0)
            assert torch.all(m.running_mean == 0) and torch.all(m.running_var == 1)
    blocks = [m for m in s.modules() if isinstance(m, BasicBlock)]
    assert len(blocks) == 8
    assert all(torch.all(b.bn2.weight == 1) for b in blocks)          # no zero-init


def test_heads_shapes():
    feats = torch.randn(4, 128)
    assert create_vicreg_head(16, 1024)(feats).shape == (4, 1024)
    assert create_projector(16, 2048)(feats).shape == (4, config.TEACHER_DIM)


def test_random_student_features_are_finite_and_batch_independent():
    images = np.random.default_rng(0).integers(0, 256, (12, 64, 64, 3), dtype=np.uint8)
    s = create_student(16, seed=0)
    a = extract(s, images, batch_size=12)
    b = extract(s, images, batch_size=5)
    assert a.shape == (12, 128) and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
