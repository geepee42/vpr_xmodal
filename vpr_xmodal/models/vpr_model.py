"""Multimodal and cross-modal VPR models.

Multimodal (image + captions -> one descriptor):

    DINOv2 -> mask pooling -> Segment x Image -> Segment x Caption -> f_phi --+
    SigLIP2 caption embeddings ---------------------------------> g_psi ------+-> fusion -> SALAD -> p

Cross-modal (captions -> query descriptor, image -> database descriptor):

    DINOv2 -> mask pooling -> Segment x Image -> f_phi -> SALAD_img  -> p_img
    SigLIP2 caption embeddings ----------------> g_psi -> SALAD_text -> p_text

``Grounder`` holds the parts shared with Stage 1 and returns the projected region
and caption tokens that the grounding loss is computed on.

``streams`` selects the single-stream ablations of the multimodal model. ``visual``
replaces Segment x Caption with self-attention over the segment tokens and zeroes the
caption half of the fusion input. ``text`` has no image modules and zeroes the visual
half. Both are trained without Stage 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn

from .backbone import DinoV2Backbone
from .grounding import Heads, SegCapAttn, SegImgAttn, SegMLP
from .region_pool import masked_region_pool
from .salad import SALAD


@dataclass
class VPRConfig:
    variant: str = "multimodal"     # multimodal | crossmodal
    backbone: str = "dinov2_vitb14"
    num_trainable_blocks: int = 3
    pretrained: bool = True
    text_dim: int = 768
    embed_dim: int = 512
    heads: int = 12
    seg_img: str = "attn"           # attn | mlp (ablation)
    streams: str = "both"           # both | visual | text (single-stream ablations, multimodal only)
    modality_dropout: float = 0.4
    num_clusters: int = 64
    cluster_dim: int = 128
    salad_dropout: float = 0.3

    def to_dict(self):
        return asdict(self)


class Grounder(nn.Module):
    def __init__(self, cfg: VPRConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.streams in ("both", "visual", "text") and (cfg.streams == "both" or cfg.variant == "multimodal")
        self.backbone = DinoV2Backbone(cfg.backbone, cfg.num_trainable_blocks, cfg.pretrained) if cfg.streams != "text" else None
        D = self.backbone.dim if self.backbone is not None else 0
        self.seg_img = (SegImgAttn(D, cfg.heads) if cfg.seg_img == "attn" else SegMLP(D)) if D else None
        self.seg_cap = SegCapAttn(D, cfg.heads) if cfg.variant == "multimodal" and D else None
        self.heads = Heads(D, cfg.text_dim if cfg.streams != "visual" else 0, cfg.embed_dim)  # only the heads in use

    def regions(self, masks, region_valid, images=None, hidden=None):
        """Contextualised segment tokens S' (B, K, D) and their validity (B, K)."""
        patches, _ = self.backbone(images) if hidden is None else self.backbone.forward_from_cache(hidden)
        segments, cov = masked_region_pool(patches, masks)
        return self.seg_img(segments, patches), region_valid & (cov > 1e-4)

    def forward(self, masks, region_valid, cap_emb, cap_valid, images=None, hidden=None) -> Dict[str, torch.Tensor]:
        if self.cfg.streams == "text":
            z_t = self.heads.g_psi(cap_emb)
            return {"z_v": torch.zeros_like(z_t), "z_t": z_t, "valid": cap_valid}
        s, valid = self.regions(masks, region_valid, images, hidden)
        if self.cfg.streams == "visual":  # Segment x Caption block used as self-attention over the segments
            z_v = self.heads.f_phi(self.seg_cap(s, s, valid))
            return {"z_v": z_v, "z_t": torch.zeros_like(z_v), "valid": valid}
        if self.seg_cap is not None:
            s = self.seg_cap(s, cap_emb, cap_valid & valid)
        return {"z_v": self.heads.f_phi(s), "z_t": self.heads.g_psi(cap_emb), "valid": valid & cap_valid}


class MultimodalVPR(nn.Module):
    def __init__(self, cfg: VPRConfig):
        super().__init__()
        assert cfg.variant == "multimodal"
        self.cfg = cfg
        self.grounder = Grounder(cfg)
        self.fusion = nn.Linear(2 * cfg.embed_dim, cfg.embed_dim)
        self.salad = SALAD(cfg.embed_dim, cfg.num_clusters, cfg.cluster_dim, cfg.salad_dropout)

    @property
    def out_dim(self) -> int:
        return self.salad.out_dim

    def modality_dropout(self, z_v, z_t):
        """With probability p, drop either the visual or the textual stream of an image."""
        B = z_v.shape[0]
        drop = torch.rand(B, device=z_v.device) < self.cfg.modality_dropout
        vis = torch.rand(B, device=z_v.device) < 0.5
        keep_v = ~(drop & vis)[:, None, None]
        keep_t = ~(drop & ~vis)[:, None, None]
        return z_v * keep_v, z_t * keep_t

    def forward(self, masks, region_valid, cap_emb, cap_valid, images=None) -> torch.Tensor:
        g = self.grounder(masks, region_valid, cap_emb, cap_valid, images=images)
        z_v, z_t = g["z_v"], g["z_t"]
        if self.training and self.cfg.modality_dropout > 0 and self.cfg.streams == "both":
            z_v, z_t = self.modality_dropout(z_v, z_t)
        e = self.fusion(torch.cat([z_v, z_t], dim=-1))  # segment-grounded tokens
        return self.salad(e, g["valid"])


class CrossModalVPR(nn.Module):
    def __init__(self, cfg: VPRConfig):
        super().__init__()
        assert cfg.variant == "crossmodal"
        self.cfg = cfg
        self.grounder = Grounder(cfg)
        self.salad_img = SALAD(cfg.embed_dim, cfg.num_clusters, cfg.cluster_dim, cfg.salad_dropout)
        self.salad_text = SALAD(cfg.embed_dim, cfg.num_clusters, cfg.cluster_dim, cfg.salad_dropout)

    @property
    def out_dim(self) -> int:
        return self.salad_img.out_dim

    def encode_image(self, images, masks, region_valid) -> torch.Tensor:
        s, valid = self.grounder.regions(masks, region_valid, images=images)
        return self.salad_img(self.grounder.heads.f_phi(s), valid)

    def encode_text(self, cap_emb, cap_valid) -> torch.Tensor:
        return self.salad_text(self.grounder.heads.g_psi(cap_emb), cap_valid)

    def forward(self, masks, region_valid, cap_emb, cap_valid, images=None):
        """Both descriptors of a training batch (one call, so DDP syncs every gradient)."""
        return self.encode_image(images, masks, region_valid), self.encode_text(cap_emb, cap_valid)


def build_model(cfg: VPRConfig) -> nn.Module:
    return MultimodalVPR(cfg) if cfg.variant == "multimodal" else CrossModalVPR(cfg)


def load_grounder(model: nn.Module, ckpt_path: str) -> None:
    """Initialise a VPR model's grounder (backbone blocks, Segment x Image/Caption, heads) from Stage 1."""
    state = torch.load(ckpt_path, map_location="cpu")["grounder"]
    model.grounder.load_state_dict(state)


def param_groups(model: nn.Module, lr: float, lr_backbone: float, weight_decay: float):
    """Trainable DINOv2 blocks at ``lr_backbone``, every other module at ``lr``."""
    backbone, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (backbone if "backbone." in n else rest).append(p)
    return [{"params": g, "lr": l, "weight_decay": weight_decay} for g, l in ((backbone, lr_backbone), (rest, lr)) if g]
