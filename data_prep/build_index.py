"""Build a cleaned region index (one row per segment-caption pair).

    python -m data_prep.build_index --captions_dir datasets/GSVCities/Captions/Medellin \
        --output datasets/GSVCities/RegionIndex/Medellin.csv

Scans ``<image_id>_captions_meta.json`` files under ``captions_dir`` and keeps a pair only
if both the segment and the caption pass the filters below. ``region_id`` indexes the
mask stack in ``<image_id>_filtered_masks.npz``; ``image_id`` is the image path relative
to ``captions_dir`` without extension (the plain stem for flat folders).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

META_SUFFIX = "_captions_meta.json"

# captions containing any of these are vague or describe material rather than content
VAGUE_TERMS = ("surface", "texture", "possibly", "looks like", "pattern", "shape", "object", "design")


def caption_ok(caption: str, min_words: int, drop_terms=()) -> bool:
    c = str(caption).lower()
    return len(c.split()) >= min_words and not any(t in c for t in (*VAGUE_TERMS, *drop_terms))


def segment_ok(area: float, bbox, min_area: int, max_aspect: float) -> bool:
    _, _, w, h = bbox  # XYWH
    return area >= min_area and max(w, h) / max(min(w, h), 1) <= max_aspect


def build_index(captions_dir: Path, min_area: int, max_aspect: float, min_words: int, drop_terms=()) -> pd.DataFrame:
    rows = []
    for meta_path in tqdm(sorted(captions_dir.rglob(f"*{META_SUFFIX}")), desc=f"Indexing {captions_dir}"):
        with open(meta_path) as f:
            meta = json.load(f)
        image_id = meta_path.relative_to(captions_dir).as_posix()[: -len(META_SUFFIX)]
        for r, (caption, area, bbox) in enumerate(zip(meta["captions"], meta["areas"], meta["bboxes"])):
            if segment_ok(area, bbox, min_area, max_aspect) and caption_ok(caption, min_words, drop_terms):
                rows.append({"image_id": image_id, "region_id": r, "caption": caption, "area": int(area)})
    return pd.DataFrame(rows, columns=["image_id", "region_id", "caption", "area"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_area", type=int, default=250)
    ap.add_argument("--max_aspect", type=float, default=10.0)
    ap.add_argument("--min_words", type=int, default=2)
    ap.add_argument("--drop_terms", nargs="*", default=[], help="extra terms to filter, e.g. signs text scene-text")
    a = ap.parse_args()

    df = build_index(Path(a.captions_dir), a.min_area, a.max_aspect, a.min_words, [t.lower() for t in a.drop_terms])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"{len(df)} pairs from {df['image_id'].nunique()} images -> {a.output}")


if __name__ == "__main__":
    main()
