"""Dataset layout. Everything lives under ``VPR_DATA_ROOT`` (default ``./datasets``):

    <Dataset>/Images/<split>/...                 images (native folders for Baidu and MSLS, see IMAGE_SUBDIRS)
    <Dataset>/Captions/<split>/<id>_filtered_masks.npz, <id>_captions_meta.json
    <Dataset>/RegionIndex/<split>.csv            cleaned segment-caption pairs (data_prep.build_index)
    <Dataset>/siglip2_text_embeddings/<split>.npy  row-aligned caption embeddings
    <Dataset>/dinov2_cache/<split>.npy           frozen-block DINOv2 tokens (scripts.cache_features)

``split`` is the city for GSVCities, ``database`` / ``query`` for the evaluation sets
and ``train`` for ScanNet. ``<id>`` is the image path relative to the split folder,
without extension.

``VPR_ANNOTATION_SET`` selects an alternative set of segment annotations next to the
default one: with ``VPR_ANNOTATION_SET=whole`` the caption, region-index, embedding and
region-cache paths become ``Captions_whole`` and so on, while images, ground truth and
the DINOv2 cache are shared. The whole-image caption ablation uses this.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("VPR_DATA_ROOT", "datasets"))
CACHE_ROOT = Path(os.environ.get("VPR_CACHE_ROOT", DATA_ROOT / "region_cache"))
ANNOTATION_SET = os.environ.get("VPR_ANNOTATION_SET", "")


def _annotated(name: str) -> str:
    return f"{name}_{ANNOTATION_SET}" if ANNOTATION_SET else name

MASK_SUFFIX = "_filtered_masks.npz"

GSV_CITIES = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", "Chicago", "Lisbon",
    "London", "LosAngeles", "Madrid", "Medellin", "Melbourne", "MexicoCity", "Miami",
    "Minneapolis", "Osaka", "OSL", "Phoenix", "PRG", "PRS", "Rome", "TRT", "WashingtonDC",
]

IMAGE_SUBDIRS = {
    ("Baidu", "database"): "training_images_undistort",
    ("Baidu", "query"): "query_images_undistort",
    ("MSLS", "val"): "train_val",
}


def images_dir(dataset: str, split: str) -> Path:
    sub = IMAGE_SUBDIRS.get((dataset, split))
    return DATA_ROOT / dataset / sub if sub else DATA_ROOT / dataset / "Images" / split


def captions_dir(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / _annotated("Captions") / split


def mask_path(dataset: str, split: str, image_id: str) -> Path:
    return captions_dir(dataset, split) / f"{image_id}{MASK_SUFFIX}"


def region_csv(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / _annotated("RegionIndex") / f"{split}.csv"


def text_emb_npy(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / _annotated("siglip2_text_embeddings") / f"{split}.npy"


def feature_cache(dataset: str, split: str) -> Path:
    return DATA_ROOT / dataset / "dinov2_cache" / f"{split}.npy"


def region_cache(dataset: str, split: str) -> Path:
    return CACHE_ROOT / f"{_annotated(f'{dataset}_{split}')}.npz"


def gsv_dataframe_csv(city: str) -> Path:
    return DATA_ROOT / "GSVCities" / "Dataframes" / f"{city}.csv"


def pitts_struct_mat(which: str) -> Path:  # NetVLAD split files, e.g. pitts30k_test.mat
    return DATA_ROOT / "Pittsburgh" / f"{which}.mat"


def msls_val_file(name: str) -> Path:  # msls_val_{dbImages,qImages,qIdx,pIdx}.npy
    return DATA_ROOT / "MSLS" / f"msls_val_{name}.npy"


def baidu_gt_dir(split: str) -> Path:
    return DATA_ROOT / "Baidu" / ("training_gt" if split == "database" else "query_gt")


def scannet_pairs(scene: str) -> Path:
    return DATA_ROOT / "ScanNet" / "Pairs" / f"{scene}.npz"
