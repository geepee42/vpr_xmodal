"""Stage 1: contrastive segment-caption grounding.

    accelerate launch -m vpr_xmodal.scripts.train_grounding --variant multimodal \
        --cities all --scannet --cached --out runs/grounding_mm
    accelerate launch -m vpr_xmodal.scripts.train_grounding --variant crossmodal --cities all --out runs/grounding_xm

Projected region tokens f_phi(S'') (S' for the cross-modal variant) and caption tokens
g_psi(T) are trained with the Multi-Similarity loss on hierarchically mined pairs. Each
process pads its segments to a fixed count so the batch can be gathered across GPUs;
padding is never mined, and the loss is rescaled so it averages over real segments only.
"""
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .. import paths as P
from ..data.common import collate_samples
from ..data.gsv_cities import GSVImageDataset
from ..data.scannet import ScanNetRegionDataset
from ..models import PAD, Grounder, HierarchicalMiner, VPRConfig, encode_labels, param_groups
from .common import Meter, add_common_args, cosine_with_warmup, distributed_miner, make_accelerator, ms_loss, save

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_grounding")

PAIR_STRIDE = 10**8  # pair ids are unique per process before gathering


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="multimodal", choices=["multimodal", "crossmodal"])
    ap.add_argument("--cities", nargs="+", default=["all"])
    ap.add_argument("--scannet", action="store_true", help="add ScanNet segments (multimodal variant)")
    ap.add_argument("--cached", action="store_true", help="read frozen-block DINOv2 tokens from scripts.cache_features")
    ap.add_argument("--batch_size", type=int, default=128, help="images per process and step")
    ap.add_argument("--seg_img", default="attn", choices=["attn", "mlp"], help="mlp: ablation without Segment x Image")
    ap.add_argument("--mining", default="full", choices=["random", "coarse", "hierarchical", "full"])
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_weak", type=int, default=4)
    ap.add_argument("--tau_p", type=float, default=0.8)
    ap.add_argument("--tau_n", type=float, default=0.5)
    ap.add_argument("--top_m", type=int, default=16)
    ap.add_argument("--ms_alpha", type=float, default=2.0)
    ap.add_argument("--ms_beta", type=float, default=50.0)
    ap.add_argument("--ms_base", type=float, default=0.5)
    add_common_args(ap, epochs=5, lr=3e-4, lr_backbone=1e-5, weight_decay=0.05, warmup=5000)
    a = ap.parse_args()
    if a.variant == "crossmodal" and a.scannet:
        ap.error("the cross-modal variant is trained on GSV-Cities only")
    return a


def segment_batch(out, clusters, process_index: int, pad_to: int):
    """Valid region and caption embeddings with packed labels, padded to ``pad_to`` rows each."""
    v = out["valid"]
    z_v, z_t, cl = out["z_v"][v], out["z_t"][v], clusters[v]
    assert (cl >= 0).all(), "training segments need coarse/fine pseudo-labels (data_prep.cluster_captions)"
    pair = torch.arange(len(cl), device=cl.device) + process_index * PAIR_STRIDE
    labels = encode_labels(cl[:, 0], cl[:, 1], pair)
    n_pad = pad_to - len(labels)
    emb = torch.cat([F.pad(z_v, (0, 0, 0, n_pad)), F.pad(z_t, (0, 0, 0, n_pad))]).float()
    lab = F.pad(labels, (0, n_pad), value=PAD)
    return emb, torch.cat([lab, lab]), len(labels)


def unpad_scale(acc, n_real: int, pad_to: int) -> torch.Tensor:
    """pml's MS loss averages over every row, padding included; this factor restores the
    mean over the real rows of the batch gathered from all processes."""
    total_real = acc.gather(torch.tensor([n_real], device=acc.device)).sum().clamp_min(1)
    return pad_to * acc.num_processes / total_real


def main():
    a = parse()
    acc = make_accelerator(a)
    cities = P.GSV_CITIES if a.cities == ["all"] else a.cities
    kw = dict(image_size=tuple(a.image_size), max_regions=a.max_regions, train=True, cached=a.cached)
    datasets = [GSVImageDataset(cities, **kw)]
    if a.scannet:
        datasets.append(ScanNetRegionDataset(**kw))
    dl = DataLoader(ConcatDataset(datasets), a.batch_size, shuffle=True, num_workers=a.num_workers,
                    collate_fn=collate_samples, pin_memory=True, drop_last=True, persistent_workers=a.num_workers > 0)

    cfg = VPRConfig(variant=a.variant, seg_img=a.seg_img, text_dim=datasets[0].cap_dim)
    model = Grounder(cfg)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.lr_backbone, a.weight_decay))
    model, opt, dl = acc.prepare(model, opt, dl)
    total = a.max_steps or a.epochs * len(dl) // a.grad_accum
    sched = cosine_with_warmup(opt, a.warmup_steps, total)

    miner = distributed_miner(acc, HierarchicalMiner(a.mining, a.top_k, a.max_weak, a.tau_p, a.tau_n, a.top_m))
    loss_fn = ms_loss(acc, a.ms_alpha, a.ms_beta, a.ms_base)
    pad_to = a.batch_size * a.max_regions
    if acc.is_main_process:
        log.info("images=%d  processes=%d  steps=%d  variant=%s  mining=%s", len(dl.dataset), acc.num_processes, total, a.variant, a.mining)

    step, meter, t0 = 0, Meter(), time.time()
    for epoch in range(a.epochs):
        model.train()
        for batch in dl:
            with acc.accumulate(model):
                out = model(batch["masks"], batch["region_valid"], batch["cap_emb"], batch["cap_valid"],
                            images=batch.get("image"), hidden=batch.get("hidden"))
                emb, labels, n = segment_batch(out, batch["clusters"], acc.process_index, pad_to)
                loss = loss_fn(emb, labels, miner(emb, labels)) * unpad_scale(acc, n, pad_to)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            sched.step()
            step += 1
            meter.update(loss=loss.item() / acc.num_processes, segments=n)  # pml's wrapper scales the loss by world size
            if step % a.log_every == 0 and acc.is_main_process:
                log.info("ep %d step %d/%d  %s  lr=%.2e  %.0fs", epoch, step, total, meter, sched.get_last_lr()[0], time.time() - t0)
                meter.reset()
            if step >= total:
                break
        save(acc, Path(a.out) / "grounder.pt", grounder=acc.unwrap_model(model).state_dict(), cfg=cfg.to_dict(), args=vars(a))
        if step >= total:
            break


if __name__ == "__main__":
    main()
