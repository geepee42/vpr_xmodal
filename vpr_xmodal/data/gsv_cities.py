"""GSV-Cities with segments, caption embeddings and pseudo-labels.

Two views over the same data:
  * GSVImageDataset  flat over images, for Stage 1 grounding (place structure unused).
                     With ``cached=True`` it returns the frozen DINOv2 tokens written by
                     ``scripts.cache_features`` instead of the image.
  * GSVPlaceDataset  K images of one place per item, the GSV-Cities protocol used for
                     place-level training.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import paths as P
from .common import IMAGE_SIZE, MAX_REGIONS, FeatureCache, build_image_transform, build_region_sample, grid_hw
from .region_cache import RegionStore

log = logging.getLogger(__name__)
DATASET = "GSVCities"


def gsv_image_name(row: pd.Series, place_id: int) -> str:
    """On-disk stem of a GSV-Cities image, built from its dataframe row."""
    pl_id = str(place_id % 10**5).zfill(7)
    return (
        f"{row['city_id']}_{pl_id}_{str(row['year']).zfill(4)}_{str(row['month']).zfill(2)}_"
        f"{str(row['northdeg']).zfill(3)}_{row['lat']}_{row['lon']}_{row['panoid']}"
    )


def ready_cities(cities: Optional[Sequence[str]] = None) -> List[str]:
    out = [c for c in (cities or P.GSV_CITIES) if P.region_cache(DATASET, c).exists() and P.text_emb_npy(DATASET, c).exists()]
    for c in set(cities or P.GSV_CITIES) - set(out):
        log.warning("Skipping city %s: region cache or caption embeddings missing", c)
    if not out:
        raise RuntimeError("No usable GSV cities. Run data_prep and scripts/build_region_cache.py first.")
    return out


class _GSVIndex:
    """Image table (city, place_uid, image_id) restricted to images with segments."""

    def __init__(self, cities: Sequence[str], min_img_per_place: int):
        self.stores: Dict[str, RegionStore] = {}
        frames = []
        for ci, city in enumerate(cities):
            store = RegionStore(P.region_cache(DATASET, city), P.text_emb_npy(DATASET, city))
            self.stores[city] = store
            df = pd.read_csv(P.gsv_dataframe_csv(city))
            df["place_uid"] = df["place_id"] + ci * 10**5  # unique across cities
            df["image_id"] = [gsv_image_name(r, pid) for pid, (_, r) in zip(df["place_id"], df.iterrows())]
            df = df[df["image_id"].map(store.__contains__)].assign(city=city)
            frames.append(df[["city", "place_uid", "image_id"]])
            log.info("%s: %d images with segments (%d places)", city, len(df), df["place_uid"].nunique())
        table = pd.concat(frames, ignore_index=True)
        counts = table.groupby("place_uid")["place_uid"].transform("size")
        self.table = table[counts >= min_img_per_place].reset_index(drop=True)
        self.cap_dim = next(iter(self.stores.values())).emb_dim
        self.place_ids = self.table["place_uid"].unique()
        self._by_place = self.table.groupby("place_uid").indices


class GSVImageDataset(Dataset):
    def __init__(
        self,
        cities: Sequence[str],
        image_size: Tuple[int, int] = IMAGE_SIZE,
        max_regions: int = MAX_REGIONS,
        train: bool = True,
        hflip_p: float = 0.5,
        cached: bool = False,
        images: bool = True,
    ):
        self.cities = ready_cities(cities)
        self.index = self._build_index()
        self.grid, self.max_regions, self.train = grid_hw(image_size), max_regions, train
        self.hflip_p = hflip_p if train else 0.0
        self.select_mode = "random" if train else "largest"
        self.transform = build_image_transform(image_size, train)
        self.cap_dim = self.index.cap_dim
        self.cached, self.images = cached, images
        self.cache = {c: FeatureCache(P.feature_cache(DATASET, c)) for c in self.cities} if cached else None

    def _build_index(self):
        return _GSVIndex(self.cities, min_img_per_place=1)

    def __len__(self):
        return len(self.index.table)

    def _sample(self, row, rng) -> Dict:
        city, iid = row["city"], row["image_id"]
        image_path = P.images_dir(DATASET, city) / f"{iid}.jpg" if self.images and not self.cached else None
        s = build_region_sample(
            image_path, P.mask_path(DATASET, city, iid), iid, self.index.stores[city],
            self.transform, self.grid, self.max_regions, self.cap_dim, self.select_mode, self.hflip_p, rng,
        )
        if self.cached:
            s["hidden"] = self.cache[city][iid]
        s["label"] = torch.tensor(int(row["place_uid"]))
        return s

    def __getitem__(self, i):
        return self._sample(self.index.table.iloc[i], np.random.default_rng())


class GSVPlaceDataset(GSVImageDataset):
    """``__getitem__`` returns ``img_per_place`` images of one place, stacked on a leading axis."""

    def __init__(self, cities, img_per_place: int = 4, **kw):
        self.img_per_place = img_per_place
        super().__init__(cities, **kw)

    def _build_index(self):
        return _GSVIndex(self.cities, min_img_per_place=self.img_per_place)

    def __len__(self):
        return len(self.index.place_ids)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        rows = self.index._by_place[self.index.place_ids[i]]
        pick = rng.choice(rows, size=self.img_per_place, replace=False)
        samples = [self._sample(self.index.table.iloc[r], rng) for r in pick]
        return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
