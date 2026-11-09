"""Per-image region index cache.

The RegionIndex CSVs are large (up to ~1 GB) because they hold the caption text.
At train time we only need, per image, the kept region ids (which index the mask
stack), the row index into the SigLIP2 embedding matrix, the region area and, for the
training corpora, the coarse/fine pseudo-labels. ``build_region_cache`` reads just
those columns once and stores them in a compact CSR-like .npz; ``RegionStore`` serves
lookups from it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def build_region_cache(csv_path: Path, out_npz: Path, chunksize: int = 500_000) -> Path:
    csv_path, out_npz = Path(csv_path), Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    has_clusters = {"coarse_id", "fine_id"} <= set(pd.read_csv(csv_path, nrows=0).columns)
    cols = ["image_id", "region_id", "area"] + (["coarse_id", "fine_id"] if has_clusters else [])
    parts = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=cols,
        dtype={"image_id": str, "region_id": np.int32, "area": np.int64},
        chunksize=chunksize,
    ):
        # pandas keeps a running RangeIndex across chunks -> global CSV row == embedding row
        chunk = chunk.assign(row=chunk.index.values.astype(np.int64))
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    n_rows = len(df)

    # stable sort keeps the original (embedding) row order inside each image group
    df = df.sort_values(["image_id", "row"], kind="stable")
    image_ids, starts = np.unique(df["image_id"].to_numpy(), return_index=True)
    offsets = np.append(starts, n_rows).astype(np.int64)

    extra = {k: df[k].to_numpy(np.int32) for k in ("coarse_id", "fine_id")} if has_clusters else {}
    np.savez(
        out_npz,
        image_ids=image_ids.astype(str),
        offsets=offsets,
        region_ids=df["region_id"].to_numpy(np.int32),
        emb_rows=df["row"].to_numpy(np.int64),
        areas=df["area"].to_numpy(np.int64),
        n_rows=np.int64(n_rows),
        **extra,
    )
    return out_npz


class RegionStore:
    """Lookup: image_id -> (region_ids, embedding_rows, areas) + memory-mapped embeddings."""

    def __init__(self, cache_npz: Path, emb_npy: Optional[Path] = None):
        z = np.load(cache_npz, allow_pickle=False)
        self.image_ids: np.ndarray = z["image_ids"]
        self.offsets: np.ndarray = z["offsets"]
        self.region_ids: np.ndarray = z["region_ids"]
        self.emb_rows: np.ndarray = z["emb_rows"]
        self.areas: np.ndarray = z["areas"]
        self.clusters: Optional[np.ndarray] = np.stack([z["coarse_id"], z["fine_id"]], 1) if "coarse_id" in z.files else None
        self.n_rows = int(z["n_rows"])
        self._index: Dict[str, int] = {iid: i for i, iid in enumerate(self.image_ids.tolist())}

        self.emb: Optional[np.memmap] = None
        self.emb_dim = 0
        if emb_npy is not None:
            self.emb = np.load(emb_npy, mmap_mode="r")
            if self.emb.shape[0] != self.n_rows:
                raise ValueError(
                    f"{emb_npy} has {self.emb.shape[0]} rows but {cache_npz} indexes {self.n_rows} CSV rows"
                )
            self.emb_dim = int(self.emb.shape[1])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._index

    def regions(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = self._index[image_id]
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.region_ids[s:e], self.emb_rows[s:e], self.areas[s:e]

    def cluster_ids(self, image_id: str) -> np.ndarray:
        """(R, 2) coarse/fine pseudo-labels, in the same order as ``regions``."""
        i = self._index[image_id]
        return self.clusters[self.offsets[i] : self.offsets[i + 1]]

    def embeddings(self, rows: np.ndarray) -> np.ndarray:
        if self.emb is None:
            raise RuntimeError("RegionStore was built without an embedding matrix")
        # fancy indexing on a memmap materialises just those rows
        return np.ascontiguousarray(self.emb[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
