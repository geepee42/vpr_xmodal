"""Baidu Mall (indoor). Camera centres come from the ``.camera`` ground-truth files; a
database image is correct for a query when its centre lies within 10 m."""
from __future__ import annotations

import numpy as np

from .. import paths as P
from .eval_base import EvalDataset, radius_positives

DATASET = "Baidu"


def camera_centre(path) -> np.ndarray:
    """Centre of projection (x, y, z), stored on the second-to-last line of a ``.camera`` file."""
    with open(path) as f:
        return np.array(f.readlines()[-2].split()[:3], dtype=np.float64)


def baidu(dist_thresh: float = 10.0, **kw) -> EvalDataset:
    items, centres = [], {}
    for split in ("database", "query"):
        paths = sorted(P.images_dir(DATASET, split).glob("*.jpg"))
        items += [(split, p, p.stem) for p in paths]
        centres[split] = np.stack([camera_centre(P.baidu_gt_dir(split) / f"{p.stem}.camera") for p in paths])
    positives = radius_positives(centres["database"], centres["query"], dist_thresh)
    return EvalDataset(DATASET, items, len(centres["database"]), positives, **kw)
