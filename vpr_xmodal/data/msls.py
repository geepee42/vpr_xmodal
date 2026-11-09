"""MSLS-Val, as a whole or separately for San Francisco (``sf``) and Copenhagen (``cph``).

Uses the standard ground-truth files ``msls_val_{dbImages,qImages,qIdx,pIdx}.npy``
(database list, query list, queries with ground truth, positives per query). The whole
set matches all 740 queries against the full database; positives never cross cities,
so a single city is evaluated against its own database.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .. import paths as P
from .eval_base import EvalDataset

DATASET, SPLIT = "MSLS", "val"
CITIES = ("sf", "cph")


def _rel(entry: str) -> str:
    """'train_val/sf/database/images/x.jpg' -> 'sf/database/images/x'."""
    return entry.split("/", 1)[1].rsplit(".", 1)[0]


def msls(city: Optional[str] = None, **kw) -> EvalDataset:
    """``city=None``: the whole MSLS-Val set; ``"sf"`` / ``"cph"``: one city."""
    assert city is None or city in CITIES, city
    db = np.load(P.msls_val_file("dbImages"))
    q_all = np.load(P.msls_val_file("qImages"))
    q_idx = np.load(P.msls_val_file("qIdx"))
    p_idx = np.load(P.msls_val_file("pIdx"), allow_pickle=True)

    in_city = lambda e: city is None or e.split("/")[1] == city
    db_rows = [i for i, e in enumerate(db) if in_city(e)]
    remap = {old: new for new, old in enumerate(db_rows)}
    queries = [(q_all[qi], np.array([remap[j] for j in p], dtype=np.int64)) for qi, p in zip(q_idx, p_idx) if in_city(q_all[qi])]

    root = P.images_dir(DATASET, SPLIT)
    items = [(SPLIT, root / f"{_rel(db[i])}.jpg", _rel(db[i])) for i in db_rows]
    items += [(SPLIT, root / f"{_rel(e)}.jpg", _rel(e)) for e, _ in queries]
    return EvalDataset(DATASET, items, len(db_rows), [p for _, p in queries], **kw)
