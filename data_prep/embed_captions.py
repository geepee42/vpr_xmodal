"""Embed region-index captions with the frozen SigLIP2 text encoder.

    python -m data_prep.embed_captions --csv datasets/GSVCities/RegionIndex/Medellin.csv \
        --output datasets/GSVCities/siglip2_text_embeddings/Medellin.npy

Row ``i`` of the output (float16, L2-normalised, N x 768) is the caption in row ``i``
of the CSV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vpr_xmodal.text_encoder import SigLIPTextEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    captions = pd.read_csv(a.csv, usecols=["caption"])["caption"].astype(str).tolist()
    encoder = SigLIPTextEncoder(device=a.device)
    chunks = [encoder(captions[i : i + a.batch_size]) for i in tqdm(range(0, len(captions), a.batch_size))]
    emb = torch.cat(chunks).numpy().astype(np.float16) if chunks else np.zeros((0, 768), np.float16)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, emb)
    print(f"{emb.shape[0]} captions -> {a.output}")


if __name__ == "__main__":
    main()
