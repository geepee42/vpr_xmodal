"""AmsterTime: archival photos (queries) paired 1:1 with present-day images (database) by
file stem. Only the paired database image is correct for a query."""
from __future__ import annotations

import numpy as np

from .. import paths as P
from .eval_base import EvalDataset

DATASET = "AmsterTime"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _files(split: str):
    return {p.stem: p for p in sorted(P.images_dir(DATASET, split).iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def amstertime(**kw) -> EvalDataset:
    db, q = _files("database"), _files("query")
    db_stems = sorted(db)
    q_stems = sorted(s for s in q if s in db)  # queries without a partner have no ground truth
    row = {s: i for i, s in enumerate(db_stems)}
    items = [("database", db[s], s) for s in db_stems] + [("query", q[s], s) for s in q_stems]
    positives = [np.array([row[s]], dtype=np.int64) for s in q_stems]
    return EvalDataset(DATASET, items, len(db_stems), positives, **kw)
