"""Build the compact per-image region caches from the region-index CSVs.

    python -m vpr_xmodal.scripts.build_region_cache --dataset GSVCities --splits all
    python -m vpr_xmodal.scripts.build_region_cache --dataset Pittsburgh --splits database query
    python -m vpr_xmodal.scripts.build_region_cache --dataset ScanNet --splits train
"""
import argparse
import time

import numpy as np

from .. import paths as P
from ..data.region_cache import build_region_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="GSVCities, ScanNet, Pittsburgh, MSLS, Baidu or AmsterTime")
    ap.add_argument("--splits", nargs="+", required=True, help="cities / database query / val / train; 'all' = every GSV city")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    splits = P.GSV_CITIES if a.splits == ["all"] else a.splits
    for split in splits:
        csv, npy, out = P.region_csv(a.dataset, split), P.text_emb_npy(a.dataset, split), P.region_cache(a.dataset, split)
        if out.exists() and not a.force:
            print(f"exists, skipping: {out}")
            continue
        if not csv.exists():
            print(f"missing {csv}")
            continue
        t0 = time.time()
        build_region_cache(csv, out)
        n_cache = int(np.load(out)["n_rows"])
        n_emb = np.load(npy, mmap_mode="r").shape[0] if npy.exists() else None
        status = "no embeddings" if n_emb is None else ("rows OK" if n_emb == n_cache else f"ROW MISMATCH ({n_emb} embeddings)")
        print(f"built {out} from {csv.name} ({n_cache} rows, {status}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
