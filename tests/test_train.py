import numpy as np
import pytest
import torch

from mo433 import config
from mo433.extract_features import build_encoder
from mo433.features import extract
from mo433.train import lr_factor, parse_run, train
from mo433.training import ViewBank

RNG = np.random.default_rng(0)
VIEWS = RNG.integers(0, 256, (48, 10, 64, 64, 3), dtype=np.uint8)
TARGETS = RNG.normal(size=(48, 10, 384)).astype(np.float16)


def test_parse_run(monkeypatch):
    monkeypatch.setitem(config.SELECTED, "C", (3e-3, 1024))
    assert parse_run("C-c32-d50-s1") == {"row": "C", "c": 32, "percent": 50, "seed": 1,
                                         "lr": 3e-3, "hidden": 1024, "epochs": config.EPOCHS,
                                         "tuning": False}
    trial = parse_run("B-c16-d100-s0-tune-lr1e-3-h2048")
    assert (trial["lr"], trial["hidden"], trial["epochs"]) == (1e-3, 2048, config.TUNING_EPOCHS)
    with pytest.raises(ValueError):
        parse_run("A-c16-s0")


def test_lr_schedule_warms_up_then_decays_to_zero():
    f = [lr_factor(s, 100, 5) for s in range(100)]
    assert f[0] == pytest.approx(0.2) and f[4] == pytest.approx(1.0) and f[5] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[5:], f[6:]))
    assert f[-1] == pytest.approx(0.0, abs=1e-3)


def test_b_and_c_same_budget_and_checkpoint_round_trip(tmp_path):
    bank = ViewBank(VIEWS, TARGETS)
    positions = np.arange(48)
    logs = {}
    for row in "BC":
        student, log = train(row, 16, 0, positions, bank, lr=1e-3, hidden=128, epochs=2,
                             batch_size=16, log=lambda *_: None)
        logs[row] = log
        assert all(np.isfinite(e["loss"]) for e in log["epochs"])
    assert logs["B"]["total_steps"] == logs["C"]["total_steps"] == 6

    # The saved student loads back as the same encoder through extract_features.
    (tmp_path / "runs").mkdir()
    torch.save(student.state_dict(), tmp_path / "runs" / "C-c16-d100-s0-x.pth")
    loaded = build_encoder("C-c16-d100-s0-x", root=tmp_path)
    images = VIEWS[:8, 0]
    np.testing.assert_allclose(extract(loaded, images), extract(student, images), rtol=1e-5,
                               atol=1e-6)


def test_divergence_raises_or_is_recorded():
    bank = ViewBank(VIEWS, TARGETS)
    kwargs = dict(lr=1e6, hidden=128, epochs=3, batch_size=16, log=lambda *_: None)
    with pytest.raises(RuntimeError):
        train("B", 16, 0, np.arange(48), bank, **kwargs)
    _, log = train("B", 16, 0, np.arange(48), bank, stop_if_diverged=True, **kwargs)
    assert log["diverged_at_epoch"] == len(log["epochs"]) + 1
