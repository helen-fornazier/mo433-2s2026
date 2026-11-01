"""Evaluation functions, applied identically to every encoder (spec 3.6-3.7; DECISOES 20-24).

Preprocessing: each feature vector is L2-normalized, then standardized with a StandardScaler
fitted on the features of the whole training split of that encoder (no labels needed), and
never on the test split. The same fitted preprocessing is applied to the labeled subsets, the
test features and the diagnostics.

This module holds the parts needed for the sanity and collapse checks (step E9): the k-NN
probe, the per-dimension standard deviation of the L2-normalized embeddings, and k-means scored
with NMI and ARI. The SVM probes are added in step E22.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from mo433 import config
from mo433.label_subsets import knn_k


def fit_preprocessor(train_feats):
    """L2 normalization followed by a StandardScaler fitted on ``train_feats`` (training split)."""
    return make_pipeline(Normalizer(norm="l2"), StandardScaler()).fit(train_feats)


def knn_accuracy(train_x, train_y, test_x, test_y) -> dict:
    """k-NN probe: cosine metric, uniform votes, k = min(20, smallest class in the subset)."""
    k = knn_k(train_y)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform", n_jobs=-1)
    clf.fit(train_x, train_y)
    return {"accuracy": float((clf.predict(test_x) == test_y).mean()), "k": k}


def embedding_std(feats) -> float:
    """Mean over dimensions of the standard deviation of the L2-normalized embeddings.

    Close to 0 means collapse (all embeddings point the same way). For reference, an isotropic
    distribution on the unit sphere in d dimensions gives about 1/sqrt(d).
    """
    z = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    return float(z.std(axis=0).mean())


def kmeans_scores(x, y, n_clusters: int = config.NUM_CLASSES) -> dict:
    """k-means (k = number of classes) on preprocessed features, scored against the true labels.
    Labels are used only to score, never to cluster."""
    km = KMeans(n_clusters=n_clusters, n_init=config.KMEANS_N_INIT, random_state=config.KMEANS_SEED)
    pred = km.fit_predict(x)
    return {"nmi": float(normalized_mutual_info_score(y, pred)),
            "ari": float(adjusted_rand_score(y, pred))}


def sanity_diagnostics(train_feats, y_train, test_feats, y_test, label_subsets) -> dict:
    """The sanity / collapse check used for the teacher (spec 3.3) and every student (3.4):
    k-NN accuracy at 100% and 1% of the labels (1%: mean over the draws), embedding std, and
    k-means NMI/ARI on the test split. ``label_subsets`` as in label_subsets.pkl."""
    pre = fit_preprocessor(train_feats)
    tr, te = pre.transform(train_feats), pre.transform(test_feats)
    knn_100 = knn_accuracy(tr, y_train, te, y_test)
    knn_1 = [knn_accuracy(tr[idx], y_train[idx], te, y_test)
             for (frac, draw), idx in sorted(label_subsets.items()) if frac == 0.01]
    return {
        "feature_dim": int(train_feats.shape[1]),
        "knn_100": knn_100["accuracy"],
        "knn_1_mean": float(np.mean([r["accuracy"] for r in knn_1])),
        "knn_1_per_draw": [r["accuracy"] for r in knn_1],
        "knn_k": {"100%": knn_100["k"], "1%": knn_1[0]["k"]},
        "embedding_std": embedding_std(test_feats),
        "embedding_std_isotropic_ref": float(1 / np.sqrt(train_feats.shape[1])),
        **{f"kmeans_{k}": v for k, v in kmeans_scores(te, y_test).items()},
    }
