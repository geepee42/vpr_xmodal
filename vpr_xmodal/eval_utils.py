"""Recall@K for place recognition: a query counts as correct if any of its top-K database matches is a positive."""
from typing import Dict, Sequence

import numpy as np
import torch


def recall_at_k(db: torch.Tensor, q: torch.Tensor, positives: Sequence[np.ndarray], ks=(1, 5, 10, 20)) -> Dict[int, float]:
    """db (Ndb, D), q (Nq, D) L2-normalised descriptors; positives[i] = db indices that are correct for query i."""
    import faiss

    db_np = np.ascontiguousarray(db.float().cpu().numpy())
    q_np = np.ascontiguousarray(q.float().cpu().numpy())
    index = faiss.IndexFlatIP(db_np.shape[1])
    index.add(db_np)
    _, pred = index.search(q_np, max(ks))
    correct = np.zeros(len(ks))
    n_eval = 0
    for qi, p in enumerate(pred):
        pos = positives[qi]
        if len(pos) == 0:
            continue
        n_eval += 1
        hits = np.isin(p, pos)
        for i, k in enumerate(ks):
            if hits[:k].any():
                correct[i:] += 1
                break
    return {k: float(c / max(n_eval, 1)) for k, c in zip(ks, correct)}


def format_recalls(r: Dict[int, float]) -> str:
    return "  ".join(f"R@{k}={100*v:5.2f}" for k, v in r.items())
