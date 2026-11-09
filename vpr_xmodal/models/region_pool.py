"""Mask-weighted average pooling of DINOv2 patch tokens into segment features."""
import torch


def masked_region_pool(patches: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6):
    """Soft masked average pooling of patch tokens.

    patches: (B, N, C) with N = gh*gw
    masks:   (B, M, gh, gw) coverage weights in [0, 1]
    returns  regions (B, M, C), coverage (B, M) = total mask mass on the grid
    """
    w = masks.flatten(2).to(patches.dtype)  # (B, M, N)
    cov = w.sum(-1)
    regions = torch.einsum("bmn,bnc->bmc", w, patches) / (cov + eps).unsqueeze(-1)
    return regions, cov
