"""Generate class-agnostic segments with the MobileSAM Automatic Mask Generator.

    python -m data_prep.segment --img_dir datasets/GSVCities/Images/Medellin \
        --out_dir datasets/GSVCities/Segments/Medellin --checkpoint weights/mobile_sam.pt

Walks ``img_dir`` recursively and mirrors its sub-folders into ``out_dir``. For every
image ``<rel>/<stem>.jpg`` it writes ``<rel>/<stem>.npz`` (boolean masks, M x H x W)
and ``<rel>/<stem>.json`` (XYWH boxes and areas). Images already processed are skipped.
``--whole_image`` writes one mask covering the whole image instead, for the
whole-image caption ablation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

AMG_PARAMS = dict(
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.8,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
)


def build_mask_generator(checkpoint: str, device: str):
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["vit_t"](checkpoint=checkpoint).eval().requires_grad_(False).to(device)
    return SamAutomaticMaskGenerator(model=sam, **AMG_PARAMS)


def whole_image_segment(h: int, w: int):
    return [{"segmentation": np.ones((h, w), bool), "bbox": [0, 0, w, h], "area": h * w}]


def save_segments(segments, hw, mask_path: Path, json_path: Path):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.array([s["segmentation"] for s in segments], dtype=bool).reshape(len(segments), *hw)
    np.savez_compressed(mask_path, masks=masks)
    with open(json_path, "w") as f:
        json.dump({"bboxes": [s["bbox"] for s in segments], "areas": [s["area"] for s in segments]}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="mobile_sam.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--whole_image", action="store_true", help="one mask covering the whole image (ablation)")
    a = ap.parse_args()

    img_dir, out_dir = Path(a.img_dir), Path(a.out_dir)
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    mask_generator = None if a.whole_image else build_mask_generator(a.checkpoint, a.device)

    for img_path in tqdm(images, desc=f"Segmenting {img_dir}"):
        mask_path = (out_dir / img_path.relative_to(img_dir)).with_suffix(".npz")
        json_path = mask_path.with_suffix(".json")
        if mask_path.exists() and json_path.exists():
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        if mask_generator is None:
            segments = whole_image_segment(*image.shape[:2])
        else:
            with torch.inference_mode():
                segments = mask_generator.generate(image)
        save_segments(segments, image.shape[:2], mask_path, json_path)


if __name__ == "__main__":
    main()
