"""Shared sample construction: image + region masks + caption embeddings.

A *sample* is one image with up to ``max_regions`` regions:
    image        (3, H, W)        normalised tensor
    masks        (M, gh, gw)      soft coverage of each region on the DINOv2 patch grid
    region_valid (M,)             bool, False for padding / dropped regions
    cap_emb      (M, D_text)      frozen SigLIP2 caption embeddings (zeros for padding)
    cap_valid    (M,)             bool
    clusters     (M, 2)           coarse/fine pseudo-labels (-1 for padding or unlabelled sets)
Masks are downsampled straight to the patch grid with area interpolation, so the
model never sees full-resolution masks (memory) and the pooling weights are
fractional coverage rather than hard 0/1 (less aliasing on small regions).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile

from .region_cache import RegionStore

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14
IMAGE_SIZE = (224, 224)
MAX_REGIONS = 32  # K_max: random subset while training, largest segments at evaluation


def grid_hw(image_size: Tuple[int, int]) -> Tuple[int, int]:
    h, w = image_size
    assert h % PATCH == 0 and w % PATCH == 0, f"image size {image_size} must be divisible by {PATCH}"
    return h // PATCH, w // PATCH


def build_image_transform(image_size: Tuple[int, int], train: bool) -> T.Compose:
    ops = [T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)]
    if train:
        # photometric only: geometric augmentation would have to be applied to the masks too
        ops.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8))
        ops.append(T.RandomGrayscale(p=0.05))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_masks_npz(path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        key = "masks" if "masks" in z.files else z.files[0]
        m = z[key]
    if m.ndim == 2:
        m = m[None]
    return m


def select_regions(
    areas: np.ndarray, max_regions: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Indices (into the per-image region list) to keep. 'largest' is deterministic
    (eval); 'random' subsamples uniformly (train-time augmentation)."""
    n = len(areas)
    if n <= max_regions:
        return np.arange(n)
    if mode == "largest":
        return np.argsort(-areas, kind="stable")[:max_regions]
    if mode == "random":
        return np.sort(rng.choice(n, size=max_regions, replace=False))
    raise ValueError(mode)


def masks_to_grid(masks: np.ndarray, grid: Tuple[int, int]) -> torch.Tensor:
    """(M, H, W) bool/uint8 -> (M, gh, gw) float coverage in [0, 1]."""
    if masks.shape[0] == 0:
        return torch.zeros((0, *grid), dtype=torch.float32)
    m = torch.from_numpy(np.ascontiguousarray(masks)).to(torch.float32).unsqueeze(0)  # (1, M, H, W)
    m = F.interpolate(m, size=grid, mode="area")
    return m.squeeze(0)


def build_region_sample(
    image_path,
    mask_path,
    image_id: str,
    store: Optional[RegionStore],
    transform: Optional[T.Compose],
    grid: Tuple[int, int],
    max_regions: int,
    cap_dim: int,
    select_mode: str = "largest",
    hflip_p: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """One image with up to ``max_regions`` segments. ``image_path=None`` skips the image
    (cached DINOv2 tokens are used instead), which also disables the joint flip."""
    rng = rng if rng is not None else np.random.default_rng()
    image = transform(load_image(image_path)) if image_path is not None else None

    masks_grid = torch.zeros((max_regions, *grid), dtype=torch.float32)
    cap_emb = torch.zeros((max_regions, cap_dim), dtype=torch.float32)
    valid = torch.zeros(max_regions, dtype=torch.bool)
    clusters = torch.full((max_regions, 2), -1, dtype=torch.long)
    n_kept = 0

    if store is not None and image_id in store and mask_path is not None and mask_path.exists():
        region_ids, emb_rows, areas = store.regions(image_id)
        keep = select_regions(areas, max_regions, select_mode, rng)
        all_masks = load_masks_npz(mask_path)
        keep = keep[region_ids[keep] < all_masks.shape[0]]  # guards against stale indices
        n_kept = len(keep)
        if n_kept:
            masks_grid[:n_kept] = masks_to_grid(all_masks[region_ids[keep]], grid)
            cap_emb[:n_kept] = torch.from_numpy(store.embeddings(emb_rows[keep]))
            valid[:n_kept] = True
            if store.clusters is not None:
                clusters[:n_kept] = torch.from_numpy(store.cluster_ids(image_id)[keep].astype(np.int64))

    if image is not None and hflip_p > 0 and rng.random() < hflip_p:
        image = torch.flip(image, dims=[-1])
        masks_grid = torch.flip(masks_grid, dims=[-1])

    sample = {
        "masks": masks_grid,
        "region_valid": valid,
        "cap_emb": cap_emb,
        "cap_valid": valid.clone(),
        "clusters": clusters,
        "n_regions": torch.tensor(n_kept),
    }
    if image is not None:
        sample["image"] = image
    return sample


class FeatureCache:
    """Frozen-block DINOv2 tokens written by ``scripts.cache_features``, looked up by image id."""

    def __init__(self, npy_path):
        self.tokens = np.load(npy_path, mmap_mode="r")
        self.row = {iid: i for i, iid in enumerate(open(f"{npy_path}.ids.txt").read().split())}

    def __getitem__(self, image_id: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.tokens[self.row[image_id]], dtype=np.float32))


def collate_samples(samples: Sequence[Dict]) -> Dict:
    out = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out


def flatten_places(batch: Dict) -> Dict:
    """(P, K, ...) place batches -> (P*K, ...)."""
    return {k: v.flatten(0, 1) if torch.is_tensor(v) and v.dim() >= 2 else v for k, v in batch.items()}
