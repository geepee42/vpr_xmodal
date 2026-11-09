"""Hierarchical positive/negative mining for segment-caption grounding.

Regions and captions are mined together. Every item carries its coarse and fine
pseudo-label and the id of its segment-caption pair, packed into one integer label
(``encode_labels``) so the miner also works through pml's distributed wrappers.

For an anchor i (strategy ``full``):
  strong positives  same coarse and fine cluster, top-K most similar
  weak positives    same coarse, different fine, similarity > tau_p, at most ``max_weak``
  hard negatives    same coarse, different fine, tau_n < similarity <= tau_p
  semi-hard negs    other coarse clusters, top-M most similar
The item's own pair (its caption, or its region) is always a positive. Items labelled
``PAD`` (-1) are padding that keeps per-GPU batch sizes equal; they are never mined.

Ablation strategies, each adding one component to the previous one:
  random        own pair positive, every other item negative
  coarse        top-K of the same coarse cluster positive, top-M of other coarse clusters negative
  hierarchical  top-K of the same fine cluster positive; every same-coarse, different-fine
                item and the top-M of other coarse clusters negative
  full          hierarchical plus the thresholds tau_p and tau_n (above)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_metric_learning.miners import BaseMiner

PAD = -1
PAIR_BASE = 10**10   # room for pair ids
FINE_BASE = 1000     # room for fine ids inside a coarse cluster


def encode_labels(coarse: torch.Tensor, fine: torch.Tensor, pair_id: torch.Tensor) -> torch.Tensor:
    return (coarse.long() * FINE_BASE + fine.long()) * PAIR_BASE + pair_id.long()


def decode_labels(labels: torch.Tensor):
    cluster, pair_id = labels // PAIR_BASE, labels % PAIR_BASE
    return cluster // FINE_BASE, cluster, pair_id  # coarse, (coarse, fine), pair


def topk_mask(sim: torch.Tensor, cand: torch.Tensor, k: int) -> torch.Tensor:
    """Keep, per row, the k most similar candidates."""
    k = min(k, sim.shape[1])
    if k <= 0:
        return torch.zeros_like(cand)
    s = sim.masked_fill(~cand, float("-inf"))
    idx = s.topk(k, dim=1).indices
    keep = torch.zeros_like(cand).scatter_(1, idx, True)
    return keep & cand


class HierarchicalMiner(BaseMiner):
    def __init__(self, strategy: str = "full", top_k: int = 8, max_weak: int = 4, tau_p: float = 0.8,
                 tau_n: float = 0.5, top_m: int = 16, **kwargs):
        super().__init__(**kwargs)
        assert strategy in ("random", "coarse", "hierarchical", "full"), strategy
        assert tau_p > tau_n
        self.strategy, self.top_k, self.max_weak = strategy, top_k, max_weak
        self.tau_p, self.tau_n, self.top_m = tau_p, tau_n, top_m

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        sim = F.normalize(embeddings.float(), dim=-1) @ F.normalize(ref_emb.float(), dim=-1).t()
        c_a, f_a, p_a = decode_labels(labels)
        c_r, f_r, p_r = decode_labels(ref_labels)
        same_coarse = c_a[:, None] == c_r[None]
        same_fine = f_a[:, None] == f_r[None]
        same_pair = p_a[:, None] == p_r[None]
        not_self = torch.ones_like(same_pair)
        if ref_emb is embeddings:
            not_self.fill_diagonal_(False)
        real = (labels != PAD)[:, None] & (ref_labels != PAD)[None]
        not_self &= real
        own = same_pair & not_self

        semi = topk_mask(sim, ~same_coarse & real, self.top_m)
        sibling = same_coarse & ~same_fine & real
        if self.strategy == "random":
            pos, neg = own, ~same_pair & real
        elif self.strategy == "coarse":
            pos, neg = topk_mask(sim, same_coarse & not_self, self.top_k) | own, semi
        elif self.strategy == "hierarchical":
            pos, neg = topk_mask(sim, same_fine & not_self, self.top_k) | own, sibling | semi
        else:
            strong = topk_mask(sim, same_fine & not_self, self.top_k)
            weak = topk_mask(sim, sibling & (sim > self.tau_p), self.max_weak)
            hard = sibling & (sim > self.tau_n) & (sim <= self.tau_p)
            pos, neg = strong | weak | own, hard | semi

        a1, p = pos.nonzero(as_tuple=True)
        a2, n = neg.nonzero(as_tuple=True)
        return a1, p, a2, n
