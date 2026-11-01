import numpy as np
import pytest

from mo433.dataset import map_split_lists

FILES = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]


def test_split_lists_map_to_sorted_indices():
    idx = map_split_lists(FILES, {"train": ["d.jpg", "a.jpg"], "val": ["b.jpg"], "test": ["c.jpg"]})
    assert idx["train"].tolist() == [0, 3]
    assert idx["test"].tolist() == [2]


@pytest.mark.parametrize("lists", [
    {"train": ["a.jpg", "b.jpg"], "test": ["c.jpg"]},                    # d.jpg not listed
    {"train": ["a.jpg", "b.jpg"], "test": ["c.jpg", "d.jpg", "e.jpg"]},  # e.jpg not in dataset
    {"train": ["a.jpg", "b.jpg", "c.jpg"], "test": ["c.jpg", "d.jpg"]},  # c.jpg in two lists
])
def test_split_lists_must_partition_the_dataset(lists):
    with pytest.raises(ValueError):
        map_split_lists(FILES, lists)
