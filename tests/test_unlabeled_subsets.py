import inspect

import numpy as np

from mo433.unlabeled_subsets import draw_unlabeled_subset


def test_full_and_half():
    full = draw_unlabeled_subset(100, 100)
    half = draw_unlabeled_subset(100, 50)
    assert np.array_equal(full, np.arange(100))
    assert len(half) == 50 and len(np.unique(half)) == 50
    assert np.isin(half, full).all() and (np.diff(half) > 0).all()


def test_half_is_reproducible_and_seed_dependent():
    assert np.array_equal(draw_unlabeled_subset(100, 50), draw_unlabeled_subset(100, 50))
    assert not np.array_equal(draw_unlabeled_subset(100, 50, seed=0),
                              draw_unlabeled_subset(100, 50, seed=1))


def test_draw_cannot_receive_labels():
    assert list(inspect.signature(draw_unlabeled_subset).parameters) == ["n", "percent", "seed"]
