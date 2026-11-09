"""DINOv2 backbone with partial fine-tuning; returns patch + CLS tokens.

Only the last ``num_trainable_blocks`` blocks are trained. ``frozen`` runs the blocks
before them, so their output can be cached once per image and fed to
``forward_from_cache`` during grounding.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

DINOV2_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}


class DinoV2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14", num_trainable_blocks: int = 3, pretrained: bool = True):
        super().__init__()
        assert name in DINOV2_DIMS, name
        self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained, trust_repo=True, skip_validation=True)
        self.dim = DINOV2_DIMS[name]
        self.n_frozen = len(self.model.blocks) - num_trainable_blocks

        self.model.requires_grad_(False)
        for blk in self.model.blocks[self.n_frozen:]:
            blk.requires_grad_(True)
        if num_trainable_blocks > 0:
            self.model.norm.requires_grad_(True)

    @torch.no_grad()
    def frozen(self, x: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) -> hidden tokens after the frozen blocks (B, 1 + N, C)."""
        x = self.model.prepare_tokens_with_masks(x)
        for blk in self.model.blocks[: self.n_frozen]:
            x = blk(x)
        return x

    def forward_from_cache(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden tokens (B, 1 + N, C) -> patch tokens (B, N, C), CLS token (B, C)."""
        for blk in self.model.blocks[self.n_frozen:]:
            h = blk(h)
        h = self.model.norm(h)
        return h[:, 1:], h[:, 0]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_from_cache(self.frozen(x))
