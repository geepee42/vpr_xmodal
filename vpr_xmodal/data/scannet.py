"""ScanNet with instance segments, captions and pose-derived places.

  * ScanNetRegionDataset  flat over frames, for Stage 1 grounding.
  * ScanNetPlaceDataset   pseudo-places for place-level training: groups of frames from
                          one scene that are all pairwise positives (0.3-1.0 m apart with
                          sufficient view overlap, from ``data_prep.scannet``'s pair files).
  * PlaceBatchSampler     place batches; GSV-Cities and ScanNet batches are kept separate
                          and shuffled together, so each dataset contributes in proportion
                          to its size. Two ScanNet groups from the same scene enter a step
                          only if every pair of their frames is at least ``r_neg`` metres
                          apart, so different labels are always true negatives. With several
                          processes the rule covers the batches of all of them, because the
                          loss is computed on the batch gathered from every process.
  * PlaceData             GSV-Cities places followed by ScanNet pseudo-places.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import paths as P
from .common import IMAGE_SIZE, MAX_REGIONS, FeatureCache, build_image_transform, build_region_sample, grid_hw
from .region_cache import RegionStore

DATASET, SPLIT = "ScanNet", "train"
LABEL_OFFSET = 10**8  # keeps ScanNet place labels apart from GSV-Cities place ids


class ScanNetRegionDataset(Dataset):
    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE, max_regions: int = MAX_REGIONS,
                 train: bool = True, cached: bool = False, images: bool = True):
        self.store = RegionStore(P.region_cache(DATASET, SPLIT), P.text_emb_npy(DATASET, SPLIT))
        self.image_ids: List[str] = list(self.store.image_ids)
        self.grid, self.max_regions, self.train = grid_hw(image_size), max_regions, train
        self.transform = build_image_transform(image_size, train)
        self.cap_dim = self.store.emb_dim
        self.cache = FeatureCache(P.feature_cache(DATASET, SPLIT)) if cached else None
        self.images = images

    def __len__(self):
        return len(self.image_ids)

    def sample(self, image_id: str, rng) -> Dict:
        image_path = P.images_dir(DATASET, SPLIT) / f"{image_id}.jpg" if self.images and self.cache is None else None
        s = build_region_sample(
            image_path, P.mask_path(DATASET, SPLIT, image_id), image_id, self.store, self.transform, self.grid,
            self.max_regions, self.cap_dim, "random" if self.train else "largest", 0.5 if self.train else 0.0, rng,
        )
        if self.cache is not None:
            s["hidden"] = self.cache[image_id]
        s["label"] = torch.tensor(-1)  # place label, set by ScanNetPlaceDataset
        return s

    def __getitem__(self, i):
        return self.sample(self.image_ids[i], np.random.default_rng())


def pairwise_groups(n_frames: int, pairs: np.ndarray, size: int, rng) -> List[List[int]]:
    """Greedy disjoint groups of ``size`` frames in which every two frames form a pair."""
    adj = [set() for _ in range(n_frames)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    used, groups = np.zeros(n_frames, bool), []
    for a in rng.permutation(n_frames):
        if used[a]:
            continue
        group = [a]
        for c in rng.permutation(sorted(adj[a])):
            if not used[c] and all(c in adj[g] for g in group):
                group.append(c)
                if len(group) == size:
                    break
        if len(group) == size:
            groups.append(group)
            used[group] = True
    return groups


class ScanNetPlaceDataset(Dataset):
    """Item i: ``img_per_place`` frames of pseudo-place i, stacked like GSVPlaceDataset.
    ``regroup(epoch)`` draws new groups, so frames are combined differently every epoch."""

    def __init__(self, img_per_place: int = 4, seed: int = 0, **kw):
        self.frames = ScanNetRegionDataset(**kw)
        self.img_per_place, self.seed = img_per_place, seed
        self.scenes = []  # (scene, frame ids, centres, pairs between frames with segments)
        for scene in sorted({iid.split("/")[0] for iid in self.frames.image_ids}):
            z = np.load(P.scannet_pairs(scene))
            ids = [f"{scene}/{f}" for f in z["frames"]]
            ok = np.array([iid in self.frames.store for iid in ids])  # frames whose segments survived cleaning
            self.scenes.append((scene, ids, z["centres"], z["pairs"][ok[z["pairs"]].all(1)]))
        self.regroup(0)

    def regroup(self, epoch: int):
        rng = np.random.default_rng((self.seed, epoch))
        self.groups: List[List[str]] = []
        self.group_scene: List[str] = []
        self.group_centres: List[np.ndarray] = []
        for scene, ids, centres, pairs in self.scenes:
            for g in pairwise_groups(len(ids), pairs, self.img_per_place, rng):
                self.groups.append([ids[i] for i in g])
                self.group_scene.append(scene)
                self.group_centres.append(centres[g])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        samples = [self.frames.sample(iid, rng) for iid in self.groups[i]]
        out = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
        out["label"] = torch.full((len(samples),), LABEL_OFFSET + i, dtype=torch.long)
        return out


class PlaceData(Dataset):
    """Index i < len(gsv): a GSV-Cities place; above: a ScanNet pseudo-place. Unlike
    ``ConcatDataset`` it follows the ScanNet group count, which changes on ``regroup``."""

    def __init__(self, gsv: Dataset, scannet: ScanNetPlaceDataset = None):
        self.gsv, self.scannet = gsv, scannet

    def __len__(self):
        return len(self.gsv) + (len(self.scannet) if self.scannet is not None else 0)

    def __getitem__(self, i):
        return self.gsv[i] if i < len(self.gsv) else self.scannet[i - len(self.gsv)]


class PlaceBatchSampler:
    """Batches of ``places_per_batch`` place indices over ``PlaceData(gsv, scannet)``.

    Batches come in steps of ``num_processes`` consecutive batches, one per process (the
    order in which accelerate hands them out). A step holds places of one dataset only,
    and the 6 m rule applies to the whole step. Order and ScanNet groups are redrawn each
    epoch (call ``set_epoch``) with a seed shared by all processes.
    """

    def __init__(self, n_gsv: int, scannet: ScanNetPlaceDataset = None, places_per_batch: int = 32,
                 r_neg: float = 6.0, seed: int = 0, num_processes: int = 1):
        self.n_gsv, self.scannet, self.ppb, self.r_neg, self.seed = n_gsv, scannet, places_per_batch, r_neg, seed
        self.world = num_processes
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch, self._cached = epoch, None
        if self.scannet is not None:  # data loader workers start after this, so they see the new groups
            self.scannet.regroup(epoch)

    def _fits(self, g: int, batch: Dict[str, List[int]]) -> bool:
        s = self.scannet
        for h in batch.get(s.group_scene[g], []):
            d = np.linalg.norm(s.group_centres[g][:, None] - s.group_centres[h][None], axis=-1)
            if d.min() < self.r_neg:
                return False
        return True

    def _scannet_steps(self, rng) -> List[List[int]]:
        """One pass: each group joins the first open step where it is far enough from same-scene groups."""
        done, open_batches = [], []  # open batch: (list of groups, {scene: groups})
        for g in rng.permutation(len(self.scannet)):
            for groups, by_scene in open_batches:
                if self._fits(g, by_scene):
                    break
            else:
                groups, by_scene = [], {}
                open_batches.append((groups, by_scene))
            groups.append(g)
            by_scene.setdefault(self.scannet.group_scene[g], []).append(g)
            if len(groups) == self.ppb * self.world:
                done.append([self.n_gsv + h for h in groups])
                open_batches = [b for b in open_batches if b[0] is not groups]
        return done

    def _batches(self) -> List[List[int]]:
        if self._cached is None:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm, n = rng.permutation(self.n_gsv), self.ppb * self.world
            steps = [list(perm[i : i + n]) for i in range(0, self.n_gsv - n + 1, n)]
            if self.scannet is not None:
                steps += self._scannet_steps(rng)
            steps = [steps[i] for i in rng.permutation(len(steps))]
            self._cached = [s[k : k + self.ppb] for s in steps for k in range(0, n, self.ppb)]
        return self._cached

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
