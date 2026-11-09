"""Pitts30K (NetVLAD ``.mat`` split files). A database image is correct for a query when
it lies within the split's positive radius (25 m)."""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from .. import paths as P
from .eval_base import EvalDataset, radius_positives

DATASET = "Pittsburgh"


def load_netvlad_struct(mat_path):
    raw = loadmat(mat_path)["dbStruct"][0, 0]
    unwrap = lambda arr: [arr[i, 0][0] for i in range(arr.shape[0])]
    return {
        "db_fns": unwrap(raw["dbImageFns"]),
        "q_fns": unwrap(raw["qImageFns"]),
        "utm_db": raw["utmDb"].T.astype(np.float64),
        "utm_q": raw["utmQ"].T.astype(np.float64),
        "pos_thr": float(raw["posDistThr"][0, 0]),
    }


def pitts(which: str = "pitts30k_test", **kw) -> EvalDataset:
    st = load_netvlad_struct(P.pitts_struct_mat(which))
    items = [("database", P.images_dir(DATASET, "database") / fn, fn.rsplit(".", 1)[0]) for fn in st["db_fns"]]
    items += [("query", P.images_dir(DATASET, "query") / fn, fn.rsplit(".", 1)[0]) for fn in st["q_fns"]]
    positives = radius_positives(st["utm_db"], st["utm_q"], st["pos_thr"])
    return EvalDataset(DATASET, items, len(st["db_fns"]), positives, **kw)
