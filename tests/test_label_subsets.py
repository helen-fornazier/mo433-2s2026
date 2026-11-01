import numpy as np

from mo433.label_subsets import knn_k, make_label_subsets

# 3 classes with 100, 200 and 300 examples
Y = np.repeat([0, 1, 2], [100, 200, 300])


def test_sizes_stratification_and_draws():
    subsets = make_label_subsets(Y, fractions=(0.1, 1.0), draws=(0, 1, 2))
    for draw in range(3):
        idx = subsets[(0.1, draw)]
        assert len(idx) == 60 and len(np.unique(idx)) == 60
        assert np.bincount(Y[idx]).tolist() == [10, 20, 30]
        assert np.array_equal(subsets[(1.0, draw)], np.arange(len(Y)))
    assert not np.array_equal(np.sort(subsets[(0.1, 0)]), np.sort(subsets[(0.1, 1)]))


def test_subsets_are_reproducible():
    a = make_label_subsets(Y, fractions=(0.1,), draws=(0,))
    b = make_label_subsets(Y, fractions=(0.1,), draws=(0,))
    assert np.array_equal(a[(0.1, 0)], b[(0.1, 0)])


def test_knn_k_rule():
    assert knn_k(np.repeat([0, 1], [5, 50])) == 5
    assert knn_k(np.repeat([0, 1], [500, 50])) == 20
