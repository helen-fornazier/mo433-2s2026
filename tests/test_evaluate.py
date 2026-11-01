import numpy as np

from mo433.evaluate import (embedding_std, fit_preprocessor, kmeans_scores, knn_accuracy,
                            sanity_diagnostics)

RNG = np.random.default_rng(0)


def blobs(n_per_class=60, n_classes=10, dim=16, spread=0.3):
    """Well-separated Gaussian clusters, one per class."""
    centers = RNG.normal(size=(n_classes, dim)) * 3
    y = np.repeat(np.arange(n_classes), n_per_class)
    return centers[y] + RNG.normal(scale=spread, size=(len(y), dim)), y


def test_preprocessor_is_fitted_on_train_only():
    train = RNG.normal(size=(500, 8))
    test = RNG.normal(size=(200, 8)) + 5            # shifted test split
    pre = fit_preprocessor(train)
    assert np.allclose(pre.transform(train).mean(0), 0, atol=1e-6)
    # If the scaler had been fitted on the test split, this mean would also be ~0.
    assert np.abs(pre.transform(test).mean(0)).max() > 0.5


def test_separable_features_score_high():
    x, y = blobs()
    pre = fit_preprocessor(x)
    assert knn_accuracy(pre.transform(x), y, pre.transform(x), y)["accuracy"] > 0.95
    assert kmeans_scores(pre.transform(x), y)["nmi"] > 0.9


def test_shuffled_labels_are_near_chance():
    x, y = blobs(n_per_class=100)
    pre = fit_preprocessor(x)
    xt = pre.transform(x)
    acc = knn_accuracy(xt[::2], RNG.permutation(y[::2]), xt[1::2], y[1::2])["accuracy"]
    assert acc < 0.2


def test_collapsed_features_are_detected():
    const = np.ones((300, 16)) + RNG.normal(scale=1e-9, size=(300, 16))
    y = np.repeat(np.arange(10), 30)
    assert embedding_std(const) < 1e-6
    assert kmeans_scores(fit_preprocessor(const).transform(const), y)["nmi"] < 0.1


def test_sanity_diagnostics_uses_the_1pct_subsets():
    x, y = blobs(n_per_class=100)
    subsets = {(0.01, d): np.arange(d, len(y), 100) for d in range(3)}   # 1 per class
    subsets[(1.0, 0)] = np.arange(len(y))
    out = sanity_diagnostics(x, y, x, y, subsets)
    assert out["knn_k"] == {"100%": 20, "1%": 1}
    assert len(out["knn_1_per_draw"]) == 3
