import numpy as np
import pytest
import torch

from mo433.student import create_student
from mo433.training import ViewBank, create_row, epoch_batches, steps_per_epoch, train_step

RNG = np.random.default_rng(0)
VIEWS = RNG.integers(0, 256, (40, 10, 64, 64, 3), dtype=np.uint8)
# A target that is a known function of its view: the per-channel mean, tiled to 384 dims.
TARGETS = np.tile(VIEWS.mean(axis=(2, 3)), (1, 1, 128)).astype(np.float16)


def batches(n_views, seed=0, positions=np.arange(40), batch_size=16):
    g = torch.Generator().manual_seed(seed)
    return list(epoch_batches(positions, 10, batch_size, n_views, g))


def test_b_and_c_take_the_same_steps():
    assert len(batches(1)) == len(batches(2)) == steps_per_epoch(40, 16) == 2   # 40 = 2*16 + 8


def test_each_image_once_per_epoch_and_reproducible():
    idx = torch.cat([i for i, _ in batches(1)])
    assert len(set(idx.tolist())) == len(idx) == 32
    a, b = batches(2, seed=3), batches(2, seed=3)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))
    assert not torch.equal(batches(2, seed=4)[0][0], a[0][0])


def test_row_b_views_are_distinct_and_cover_all_k():
    ks = torch.cat([k for _, k in batches(2, positions=np.arange(4000) % 40, batch_size=500)])
    assert ks.shape[1] == 2 and (ks[:, 0] != ks[:, 1]).all()
    assert set(ks.flatten().tolist()) == set(range(10))


def test_subset_positions_are_respected():
    idx = torch.cat([i for i, _ in batches(1, positions=np.arange(0, 40, 2), batch_size=4)])
    assert set(idx.tolist()) <= set(range(0, 40, 2))


def test_row_c_target_is_the_teacher_feature_of_the_same_view():
    bank = ViewBank(VIEWS, TARGETS)
    for idx, k in batches(1):
        imgs = bank.images(idx, k)                                   # (B, 3, 64, 64)
        expected = imgs.float().mean(dim=(2, 3)).repeat(1, 128)
        torch.testing.assert_close(bank.teacher(idx, k), expected, atol=0.1, rtol=1e-3)
        np.testing.assert_array_equal(imgs.permute(0, 2, 3, 1).numpy(), VIEWS[idx, k])


@pytest.mark.parametrize("row", ["B", "C"])
def test_rows_start_from_row_a_and_train(row):
    model = create_row(row, 16, 256, seed=1)
    ref = create_student(16, seed=1).state_dict()
    assert all(torch.equal(v, ref[k]) for k, v in model.student.state_dict().items())
    bank = ViewBank(VIEWS, TARGETS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    losses = [float(train_step(model, opt, scaler, bank, idx, k))
              for idx, k in batches(model.n_views, batch_size=8)]
    assert len(losses) == 5 and np.isfinite(losses).all()
