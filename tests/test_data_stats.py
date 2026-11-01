import numpy as np

from mo433.data_stats import (cross_split_duplicates, degenerate_images, exact_duplicates,
                              max_proportion_gap)


def test_exact_duplicates_and_cross_split():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (5, 4, 4, 3), dtype=np.uint8)
    images[3] = images[0]                                   # 0 and 3 are identical
    groups = exact_duplicates(images)
    assert groups == [[0, 3]]
    assert cross_split_duplicates(groups, np.array([0, 1]), np.array([3, 4])) == [[0, 3]]
    assert cross_split_duplicates(groups, np.array([0, 3]), np.array([1, 4])) == []


def test_degenerate_images():
    images = np.full((3, 4, 4, 3), 128, dtype=np.uint8)    # all constant
    images[1] = 0                                          # constant and black
    images[2, ::2] = 255                                   # not constant
    found = degenerate_images(images)
    assert found["constant"] == [0, 1]
    assert found["near_black"] == [1]


def test_max_proportion_gap_is_zero_for_identical_proportions():
    counts = {"all": {"a": 10, "b": 30}, "half": {"a": 5, "b": 15}, "skewed": {"a": 20, "b": 20}}
    gaps = max_proportion_gap(counts)
    assert gaps["half"] == 0
    assert gaps["skewed"] == 25.0
