"""SALAD (Izquierdo & Civera, CVPR 2024) over a variable number of segment tokens.

The aggregator follows the SALAD reference implementation
(https://github.com/serizba/salad, ``models/aggregators/salad.py``) without its global
token. Images are padded to a common token count and processed as one batch; padded
tokens get zero transport mass and each image's marginals use its own valid token count,
so the result equals running SALAD on every image separately.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
# (as used in SALAD), with two changes marked below for a variable number of tokens.
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0, valid=None):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    # change 1: each image uses its own valid token count; padded tokens get zero mass
    n_img = valid.sum(1, keepdim=True).to(S.dtype)  # (B, 1)
    norm = -torch.log(n_img + m)
    log_a = norm.expand(batch_size, m + 1).clone()
    log_b = norm.expand(batch_size, n).masked_fill(~valid, -math.inf)
    # change 2: dustbin mass n - m, clamped to one token when an image has fewer tokens than clusters
    log_a[:, -1:] = log_a[:, -1:] + torch.log((n_img - m).clamp_min(1))
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm.unsqueeze(-1)


class SALAD(nn.Module):
    """Locally aggregated part of SALAD: score and feature MLPs, learned dustbin, Sinkhorn."""

    def __init__(self, num_channels: int = 512, num_clusters: int = 64, cluster_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_clusters, self.cluster_dim = num_clusters, cluster_dim
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cluster_features = nn.Sequential(nn.Linear(num_channels, 512), drop, nn.ReLU(), nn.Linear(512, cluster_dim))
        self.score = nn.Sequential(nn.Linear(num_channels, 512), drop, nn.ReLU(), nn.Linear(512, num_clusters))
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def out_dim(self) -> int:
        return self.num_clusters * self.cluster_dim

    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens (B, K, C), valid (B, K) bool -> L2-normalised descriptors (B, m * l)."""
        if valid is None:
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        empty = ~valid.any(1)
        valid = valid.clone()
        valid[empty, 0] = True  # keeps Sinkhorn finite; these descriptors are zeroed below
        f = self.cluster_features(tokens).float()                     # (B, K, l)
        p = self.score(tokens).float().transpose(1, 2)                # (B, m, K)
        p = get_matching_probs(p, self.dust_bin, 3, valid=valid)      # (B, m + 1, K) log-assignment
        p = p.exp()[:, :-1].masked_fill(~valid.unsqueeze(1), 0.0)     # dustbin removed, padding zeroed
        agg = F.normalize(p @ f, dim=-1).flatten(1)                   # (B, m * l)
        return F.normalize(agg, dim=-1) * ~empty.unsqueeze(1)
