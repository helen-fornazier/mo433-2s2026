import numpy as np
import pytest

from mo433.split import make_split


def test_train_is_train_plus_val_sorted():
    lists = {"train": np.array([4, 0]), "val": np.array([2]), "test": np.array([1, 3])}
    split = make_split(lists)
    assert split["train"].tolist() == [0, 2, 4]
    assert split["test"].tolist() == [1, 3]
    assert split["train"].dtype == np.int64


def test_overlap_is_rejected():
    lists = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([2, 3])}
    with pytest.raises(ValueError):
        make_split(lists)
