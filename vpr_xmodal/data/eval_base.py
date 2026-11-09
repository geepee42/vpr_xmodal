"""Shared evaluation-set machinery.

An evaluation set is a list of database images followed by query images, each with its
segments and captions, plus the database indices that count as correct for each query.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import paths as P
from .common import IMAGE_SIZE, MAX_REGIONS, build_image_transform, build_region_sample, grid_hw
from .region_cache import RegionStore


def radius_positives(db_xy: np.ndarray, q_xy: np.ndarray, radius: float) -> List[np.ndarray]:
    """Database indices within ``radius`` (same units as the coordinates) of each query."""
    index = faiss.IndexFlatL2(db_xy.shape[1])
    index.add(np.ascontiguousarray(db_xy, dtype=np.float32))
    lims, _, ids = index.range_search(np.ascontiguousarray(q_xy, dtype=np.float32), radius**2)
    return [np.sort(ids[lims[i] : lims[i + 1]]).astype(np.int64) for i in range(len(q_xy))]


class EvalDataset(Dataset):
    """Items are ``(split, image_path, image_id)``, database first. Order: [database..., queries...]."""

    def __init__(self, dataset: str, items: Sequence[Tuple[str, object, str]], num_db: int,
                 positives: List[np.ndarray], image_size=IMAGE_SIZE, max_regions: int = MAX_REGIONS,
                 use_regions: bool = True, images: bool = True):
        self.dataset, self.items = dataset, list(items)
        self.num_db, self.num_q = num_db, len(self.items) - num_db
        self.positives, self.images = positives, images
        self.stores = {}
        if use_regions:
            for split in {s for s, _, _ in self.items}:
                self.stores[split] = RegionStore(P.region_cache(dataset, split), P.text_emb_npy(dataset, split))
        self.cap_dim = next(iter(self.stores.values())).emb_dim if self.stores else 768
        self.grid, self.max_regions = grid_hw(image_size), max_regions
        self.transform = build_image_transform(image_size, train=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        split, image_path, image_id = self.items[i]
        s = build_region_sample(
            image_path if self.images else None, P.mask_path(self.dataset, split, image_id), image_id, self.stores.get(split),
            self.transform, self.grid, self.max_regions, self.cap_dim, "largest", 0.0,
        )
        s["index"] = torch.tensor(i)
        return s
