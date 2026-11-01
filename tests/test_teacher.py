import pytest
import torch

from mo433 import config
from mo433.teacher import load_teacher
from mo433.transforms import to_tensor


@pytest.fixture(scope="module")
def teacher():
    # Random weights: same architecture and wrapper, no download needed.
    return load_teacher(pretrained=False)


def test_teacher_is_frozen_and_stays_in_eval(teacher):
    assert sum(p.requires_grad for p in teacher.parameters()) == 0
    teacher.train()
    assert not teacher.training and not teacher.vit.training


def test_teacher_output(teacher):
    x = to_tensor(torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8))
    a, b = teacher(x), teacher(x)
    assert a.shape == (2, config.TEACHER_DIM) and a.dtype == torch.float32
    assert torch.equal(a, b)
    assert not a.requires_grad
    assert teacher.vit.global_pool == "token"
