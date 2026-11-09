"""Two-level k-means over caption embeddings -> hierarchical pseudo-labels.

    python -m data_prep.cluster_captions \
        --index_dirs datasets/GSVCities/RegionIndex datasets/ScanNet/RegionIndex \
        --emb_dirs datasets/GSVCities/siglip2_text_embeddings datasets/ScanNet/siglip2_text_embeddings \
        --out datasets/clusters.npz

Fitted once on the training corpora only. Level 1 splits all training captions into
``K1`` coarse clusters; level 2 splits each coarse cluster into up to ``K2`` fine
clusters. Fine clusters with fewer than ``--min_fine_size`` members are removed and
their members reassigned to the nearest retained fine centroid of the same coarse
cluster. ``coarse_id`` and ``fine_id`` (local to the coarse cluster) are written back
into every region-index CSV.
"""
import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def kmeans(x: np.ndarray, k: int, a) -> faiss.Kmeans:
    km = faiss.Kmeans(x.shape[1], k, niter=a.niter, seed=a.seed, gpu=a.gpu,
                      max_points_per_centroid=a.max_points_per_centroid)
    km.train(x)
    return km


def assign(centroids: np.ndarray, x: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    return index.search(x, 1)[1][:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dirs", nargs="+", required=True)
    ap.add_argument("--emb_dirs", nargs="+", required=True, help="one per index dir, same order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k1", type=int, default=64)
    ap.add_argument("--k2", type=int, default=32)
    ap.add_argument("--min_fine_size", type=int, default=1000)
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--max_points_per_centroid", type=int, default=100_000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pairs = [(c, Path(e) / f"{c.stem}.npy") for i, e in zip(a.index_dirs, a.emb_dirs) for c in sorted(Path(i).glob("*.csv"))]
    csvs = [c for c, _ in pairs]
    embs = [np.load(e).astype(np.float32) for _, e in pairs]
    sizes = [len(e) for e in embs]
    x = np.concatenate(embs)
    del embs
    print(f"{len(x):,} captions from {len(csvs)} files")

    coarse_km = kmeans(x, a.k1, a)
    coarse = assign(coarse_km.centroids, x)
    fine = np.zeros(len(x), dtype=np.int64)
    fine_centroids = np.zeros((a.k1, a.k2, x.shape[1]), dtype=np.float32)
    fine_kept = np.zeros((a.k1, a.k2), dtype=bool)

    for c in range(a.k1):
        idx = np.flatnonzero(coarse == c)
        if len(idx) == 0:
            continue
        xc = x[idx]
        k = min(a.k2, len(idx))
        cent = kmeans(xc, k, a).centroids
        counts = np.bincount(assign(cent, xc), minlength=k)
        kept = np.flatnonzero(counts >= a.min_fine_size)
        if len(kept) == 0:
            kept = np.array([counts.argmax()])
        fine[idx] = kept[assign(cent[kept], xc)]
        fine_centroids[c, kept] = cent[kept]
        fine_kept[c, kept] = True
    print(f"retained {fine_kept.sum()} of {a.k1 * a.k2} fine clusters")

    np.savez(a.out, coarse_centroids=coarse_km.centroids, fine_centroids=fine_centroids, fine_kept=fine_kept)
    start = 0
    for csv, n in zip(csvs, sizes):
        df = pd.read_csv(csv)
        df["coarse_id"], df["fine_id"] = coarse[start : start + n], fine[start : start + n]
        df.to_csv(csv, index=False)
        start += n


if __name__ == "__main__":
    main()
