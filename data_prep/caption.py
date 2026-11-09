"""Caption every segment with the Describe Anything Model (DAM-3B).

    python -m data_prep.caption --img_dir datasets/GSVCities/Images/Medellin \
        --masks_dir datasets/GSVCities/Segments/Medellin \
        --captions_dir datasets/GSVCities/Captions/Medellin -b 64

Reads the ``<stem>.npz`` / ``<stem>.json`` pairs written by ``data_prep.segment``
(recursively, mirroring sub-folders), drops masks smaller than ``--min_area`` pixels and
captions the rest in batches that span images. For each image it writes
``<stem>_filtered_masks.npz`` and ``<stem>_captions_meta.json``, both sorted by area.
Weights are fetched through the Hugging Face cache (set ``HF_HOME`` to relocate it).
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DAM_PARAMS = dict(temperature=0.7, top_p=0.9, max_new_tokens=250)


def load_dam(device: str):
    from dam import DescribeAnythingModel, disable_torch_init

    disable_torch_init()
    return DescribeAnythingModel(
        model_path="nvidia/DAM-3B", conv_mode="v1", prompt_mode="full+crop", device=device,
    ).eval().requires_grad_(False)


def describe(dam, imgs, masks, query):
    """Batched DAM call; halves the batch on CUDA OOM."""
    try:
        return dam.get_description(imgs, masks, query, **DAM_PARAMS)
    except torch.cuda.OutOfMemoryError:
        if len(imgs) == 1:
            raise
        gc.collect()
        torch.cuda.empty_cache()
        mid = len(imgs) // 2
        return describe(dam, imgs[:mid], masks[:mid], query) + describe(dam, imgs[mid:], masks[mid:], query)


def find_image(img_dir: Path, rel: Path) -> Path:
    for ext in IMAGE_EXTS:
        p = img_dir / f"{rel}{ext}"  # appended, not with_suffix: GSV names contain dots
        if p.exists():
            return p
    raise FileNotFoundError(img_dir / rel)


def save_image(state, out_stem: Path):
    order = np.argsort(-state["areas"])
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out_stem}_filtered_masks.npz", masks=state["masks"][order])
    with open(f"{out_stem}_captions_meta.json", "w") as f:
        json.dump({
            "filename": out_stem.name,
            "areas": state["areas"][order].tolist(),
            "bboxes": state["bboxes"][order].tolist(),
            "captions": [state["captions"][i] for i in order],
        }, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--captions_dir", required=True)
    ap.add_argument("-b", "--batch_size", type=int, default=64)
    ap.add_argument("--min_area", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from dam.model.constants import DEFAULT_IMAGE_TOKEN

    img_dir, masks_dir, captions_dir = Path(a.img_dir), Path(a.masks_dir), Path(a.captions_dir)
    query = f"{DEFAULT_IMAGE_TOKEN}\nDescribe the highlighted region."
    dam = load_dam(a.device)

    pending = {}  # out_stem -> per-image state, until every mask has a caption
    buf = {"imgs": [], "masks": [], "keys": []}

    def flush():
        if not buf["imgs"]:
            return
        with torch.inference_mode():
            captions = describe(dam, buf["imgs"], buf["masks"], query)
        for caption, (stem, k) in zip(captions, buf["keys"]):
            pending[stem]["captions"][k] = caption
        for stem in [s for s, st in pending.items() if None not in st["captions"]]:
            save_image(pending.pop(stem), stem)
        for v in buf.values():
            v.clear()

    for mask_path in tqdm(sorted(masks_dir.rglob("*.npz")), desc=f"Captioning {masks_dir}"):
        rel = mask_path.relative_to(masks_dir).with_suffix("")
        out_stem = captions_dir / rel
        if Path(f"{out_stem}_captions_meta.json").exists() and Path(f"{out_stem}_filtered_masks.npz").exists():
            continue

        masks = np.load(mask_path)["masks"]
        with open(mask_path.with_suffix(".json")) as f:
            meta = json.load(f)
        keep = masks.sum(axis=(1, 2)) >= a.min_area
        if not keep.any():
            continue

        img = Image.open(find_image(img_dir, rel)).convert("RGB")
        state = {
            "masks": masks[keep],
            "areas": np.asarray(meta["areas"])[keep],
            "bboxes": np.asarray(meta["bboxes"])[keep],
            "captions": [None] * int(keep.sum()),
        }
        pending[out_stem] = state
        for k, m in enumerate(state["masks"]):
            buf["imgs"].append(img)
            buf["masks"].append(m)
            buf["keys"].append((out_stem, k))
            if len(buf["imgs"]) >= a.batch_size:
                flush()
    flush()


if __name__ == "__main__":
    main()
