"""Step E23: unsupervised diagnostics of one encoder on its test features (spec 3.7; DECISOES 24).

    python -m mo433.diagnostics <encoder>        e.g. A-c16-s0, B-c16-d100-s0, D

On the test-split features, preprocessed as for the probes (L2 + StandardScaler fitted on all
training features):
  k-means   k = number of classes, scored against the true labels with NMI and ARI (the same
            function and seed as the sanity checks, so the values equal sanity/<encoder>.json)
  t-SNE     2-D projection with the same parameters for every encoder (TSNE_PERPLEXITY, PCA
            initialization, TSNE_SEED), for the figures; labels are not used
Writes $MO433_DATA/diag/<encoder>_proj.npy (N_test x 2, float32, in the order of
split_idx.pkl["test"]) and then diag/<encoder>.json (the make target).
"""

import argparse
import time

import numpy as np
from sklearn.manifold import TSNE

from mo433 import config
from mo433.evaluate import fit_preprocessor, kmeans_scores
from mo433.io import atomic_path, banner, data_dir, load_pickle, save_json, write_meta

TSNE_PARAMS = {"n_components": 2, "perplexity": config.TSNE_PERPLEXITY, "init": "pca",
               "learning_rate": "auto", "random_state": config.TSNE_SEED}


def tsne_projection(x) -> np.ndarray:
    """2-D t-SNE of the preprocessed features ``x``, with the parameters fixed for all encoders."""
    return TSNE(**TSNE_PARAMS, n_jobs=-1).fit_transform(x).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoder", help="an encoder with cached features")
    args = parser.parse_args()
    root = data_dir()
    out = root / "diag" / f"{args.encoder}.json"
    banner(f"E23: diagnostics of {args.encoder}")
    labels = np.load(root / "eurosat" / "labels.npy")
    y_test = labels[load_pickle(root / "split_idx.pkl")["test"]]
    train_feats = np.load(root / "feats" / f"{args.encoder}_train.npy")
    test_feats = np.load(root / "feats" / f"{args.encoder}_test.npy")
    x = fit_preprocessor(train_feats).transform(test_feats)

    scores = kmeans_scores(x, y_test)
    start = time.perf_counter()
    proj = tsne_projection(x)
    seconds = time.perf_counter() - start
    proj_path = out.parent / f"{args.encoder}_proj.npy"
    with atomic_path(proj_path) as tmp, open(tmp, "wb") as f:
        np.save(f, proj)
    save_json({"encoder": args.encoder, "feature_dim": int(x.shape[1]),
               "kmeans_nmi": scores["nmi"], "kmeans_ari": scores["ari"],
               "tsne": TSNE_PARAMS, "tsne_seconds": round(seconds, 1)},
              out)
    write_meta(out, encoder=args.encoder, tsne=TSNE_PARAMS)
    print(f"✓ {args.encoder}: NMI {scores['nmi']:.3f}  ARI {scores['ari']:.3f}  "
          f"t-SNE {seconds:.0f} s, written to {out}")


if __name__ == "__main__":
    main()
