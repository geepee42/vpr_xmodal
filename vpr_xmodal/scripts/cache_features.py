"""Cache the output of DINOv2's frozen blocks for the grounding stage.

    python -m vpr_xmodal.scripts.cache_features --img_dir datasets/GSVCities/Images/Medellin \
        --out datasets/GSVCities/dinov2_cache/Medellin.npy

Only the last ``--num_trainable_blocks`` blocks are fine-tuned, so the earlier blocks
give the same tokens every epoch. They are stored once per image (fp16, N x (1 + P) x C)
in a memory-mapped ``.npy``, with the image ids (path relative to ``--img_dir``, no
extension) in ``<out>.ids.txt`` (row order).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.common import build_image_transform
from ..models import DinoV2Backbone

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class ImageFolder(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--num_trainable_blocks", type=int, default=3)
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.img_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    backbone = DinoV2Backbone(a.backbone, a.num_trainable_blocks).to(a.device).eval()
    n_tokens = 1 + (a.image_size[0] // 14) * (a.image_size[1] // 14)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(paths), n_tokens, backbone.dim))
    img_dir = Path(a.img_dir)
    Path(f"{out}.ids.txt").write_text("\n".join(p.relative_to(img_dir).with_suffix("").as_posix() for p in paths) + "\n")

    dl = DataLoader(ImageFolder(paths, build_image_transform(tuple(a.image_size), train=False)),
                    a.batch_size, num_workers=a.num_workers, pin_memory=True)
    i = 0
    for images in tqdm(dl, desc=f"Caching {a.img_dir}"):
        with torch.inference_mode(), torch.autocast(a.device.split(":")[0], dtype=torch.bfloat16, enabled=a.device != "cpu"):
            h = backbone.frozen(images.to(a.device))
        cache[i : i + len(h)] = h.float().cpu().numpy().astype(np.float16)
        i += len(h)
    cache.flush()


if __name__ == "__main__":
    main()
