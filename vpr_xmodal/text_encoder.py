"""Frozen SigLIP2 text encoder for segment captions.

Used by ``data_prep.embed_captions`` to precompute caption embeddings and for encoding
new captions online, so both always share the same checkpoint and preprocessing:
lowercased text, ``max_length`` padding to 64 tokens, as SigLIP2 was trained.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


class SigLIPTextEncoder:
    def __init__(self, model_name: str = "google/siglip2-base-patch16-224", device: str = "cuda", max_length: int = 64):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [t.lower() for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**enc)
            feats = feats if torch.is_tensor(feats) else feats.pooler_output  # transformers >= 5 returns an output object
            out.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, self.model.config.text_config.hidden_size)
