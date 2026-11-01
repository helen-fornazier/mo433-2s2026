import json

import pytest

from mo433 import config
from mo433.io import atomic_path, load_pickle, save_pickle, write_meta


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    out = tmp_path / "x.pkl"
    with pytest.raises(RuntimeError):
        with atomic_path(out) as tmp:
            tmp.write_text("partial")
            raise RuntimeError("interrupted")
    assert not out.exists()
    assert not (tmp_path / "x.pkl.tmp").exists()


def test_pickle_roundtrip_and_meta(tmp_path):
    out = tmp_path / "sub" / "obj.pkl"
    save_pickle({"a": [1, 2]}, out)
    assert load_pickle(out) == {"a": [1, 2]}
    write_meta(out, seed=0)
    meta = json.loads((tmp_path / "sub" / "obj.pkl.json").read_text())
    assert meta["seed"] == 0
    assert "packages" in meta["environment"]


def test_config_sizes_are_consistent():
    assert config.N_TRAIN + config.N_TEST == config.N_IMAGES
    assert set(config.SEEDS) == set(config.WIDTHS)
