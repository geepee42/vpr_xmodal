"""Segment-caption grounding blocks and projection heads.

  Segment x Image   : segment tokens attend to all image patch tokens (full transformer
                      layer: attention, residual + norm, FFN, residual + norm).
  Segment x Caption : contextualised segments attend to the image's caption embeddings
                      (Q/K/V/O projections with a residual update, no FFN).
  f_phi / g_psi     : linear heads into the shared 512-d space.

Padded captions are excluded from attention with ``key_padding_mask``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SegImgAttn(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, segments: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """segments (B, K, D), patches (B, N, D) -> contextualised segments S' (B, K, D)."""
        x = self.norm1(segments + self.attn(segments, patches, patches, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class SegMLP(nn.Module):
    """Ablation: a comparably sized MLP on the pooled segments instead of Segment x Image."""

    def __init__(self, dim: int = 768, hidden: int = 4608):  # 2 * 768 * 4608 weights, as many as Segment x Image
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, segments: torch.Tensor, patches: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.norm(segments + self.net(segments))


class SegCapAttn(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, segments: torch.Tensor, captions: torch.Tensor, cap_valid: torch.Tensor) -> torch.Tensor:
        """segments S' (B, K, D), captions T (B, K, D), cap_valid (B, K) -> S'' (B, K, D)."""
        pad = ~cap_valid
        pad[pad.all(1)] = False  # an image without captions attends to its (zero) slots instead of NaNs
        return segments + self.attn(segments, captions, captions, key_padding_mask=pad, need_weights=False)[0]


class Heads(nn.Module):
    """f_phi (visual) and g_psi (textual) projections into the shared space; a zero input
    size leaves that head out (single-stream ablations)."""

    def __init__(self, vis_dim: int = 768, text_dim: int = 768, embed_dim: int = 512):
        super().__init__()
        self.f_phi = nn.Linear(vis_dim, embed_dim) if vis_dim else None
        self.g_psi = nn.Linear(text_dim, embed_dim) if text_dim else None
