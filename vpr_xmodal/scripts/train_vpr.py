"""Stage 2: place-level training of the global descriptors.

    accelerate launch -m vpr_xmodal.scripts.train_vpr --variant multimodal --cities all --scannet \
        --grounding_ckpt runs/grounding_mm/grounder.pt --out runs/vpr_mm
    accelerate launch -m vpr_xmodal.scripts.train_vpr --variant crossmodal --cities all \
        --grounding_ckpt runs/grounding_xm/grounder.pt --out runs/vpr_xm

The grounder (DINOv2 blocks, Segment x Image / Caption, heads) starts from Stage 1; the
fusion MLP and SALAD aggregator(s) are new. Without ``--grounding_ckpt`` every module
starts fresh (DINOv2 from its pre-trained weights): the single-stream (``--streams``)
and whole-image caption ablations are trained this way. Batches hold
``places_per_batch`` places of ``img_per_place`` images. Multimodal descriptors are
trained with the Multi-Similarity loss over place labels; the cross-modal text and
image descriptors are trained against each other in both directions.
"""
import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .. import paths as P
from ..data.common import collate_samples, flatten_places
from ..data.gsv_cities import GSVPlaceDataset
from ..data.scannet import PlaceBatchSampler, PlaceData, ScanNetPlaceDataset
from ..eval_utils import format_recalls
from ..models import VPRConfig, build_model, load_grounder, param_groups
from .common import Meter, add_common_args, cosine_with_warmup, make_accelerator, ms_loss, ms_miner, save
from .evaluate import EVAL_SETS, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_vpr")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="multimodal", choices=["multimodal", "crossmodal"])
    ap.add_argument("--cities", nargs="+", default=["all"])
    ap.add_argument("--scannet", action="store_true", help="add ScanNet pseudo-places (multimodal variant)")
    ap.add_argument("--grounding_ckpt", default="", help="grounder.pt from train_grounding (same variant); omit to train without Stage 1")
    ap.add_argument("--streams", default="both", choices=["both", "visual", "text"],
                    help="visual / text: single-stream ablations (multimodal, without Stage 1)")
    ap.add_argument("--seg_img", default="attn", choices=["attn", "mlp"], help="without --grounding_ckpt only; otherwise read from it")
    ap.add_argument("--places_per_batch", type=int, default=32)
    ap.add_argument("--img_per_place", type=int, default=4)
    ap.add_argument("--modality_dropout", type=float, default=0.4)
    ap.add_argument("--num_clusters", type=int, default=64)
    ap.add_argument("--cluster_dim", type=int, default=128)
    ap.add_argument("--ms_alpha", type=float, default=1.0)
    ap.add_argument("--ms_beta", type=float, default=50.0)
    ap.add_argument("--ms_base", type=float, default=0.0)
    ap.add_argument("--miner_eps", type=float, default=0.1)
    ap.add_argument("--val_sets", nargs="*", default=["pitts30k_val"], choices=list(EVAL_SETS),
                    help="evaluated after every epoch; the first one selects best.pt")
    add_common_args(ap, epochs=4, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, warmup=2000)
    a = ap.parse_args()
    if a.variant == "crossmodal" and a.scannet:
        ap.error("the cross-modal variant is trained on GSV-Cities only")
    if a.streams != "both" and (a.variant == "crossmodal" or a.grounding_ckpt):
        ap.error("the single-stream ablations are multimodal and trained without Stage 1")
    return a


def main():
    a = parse()
    acc = make_accelerator(a)
    cities = P.GSV_CITIES if a.cities == ["all"] else a.cities
    images = a.streams != "text"
    kw = dict(image_size=tuple(a.image_size), max_regions=a.max_regions, train=True, images=images)
    gsv = GSVPlaceDataset(cities, img_per_place=a.img_per_place, **kw)
    scannet = ScanNetPlaceDataset(img_per_place=a.img_per_place, seed=a.seed, **kw) if a.scannet else None
    sampler = PlaceBatchSampler(len(gsv), scannet, a.places_per_batch, seed=a.seed, num_processes=acc.num_processes)
    dl = DataLoader(PlaceData(gsv, scannet), batch_sampler=sampler,  # workers are restarted each epoch (not persistent)
                    num_workers=a.num_workers, collate_fn=collate_samples, pin_memory=True)

    seg_img = a.seg_img
    if a.grounding_ckpt:
        stage1 = torch.load(a.grounding_ckpt, map_location="cpu")["cfg"]  # grounder architecture comes from Stage 1
        if stage1["variant"] != a.variant:
            raise SystemExit(f"--grounding_ckpt was trained for the {stage1['variant']} variant, not {a.variant}")
        seg_img = stage1["seg_img"]
    cfg = VPRConfig(variant=a.variant, seg_img=seg_img, streams=a.streams, text_dim=gsv.cap_dim,
                    modality_dropout=a.modality_dropout, num_clusters=a.num_clusters, cluster_dim=a.cluster_dim)
    model = build_model(cfg)
    if a.grounding_ckpt:
        load_grounder(model, a.grounding_ckpt)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.lr_backbone, a.weight_decay))
    model, opt, dl = acc.prepare(model, opt, dl)
    total = a.max_steps or a.epochs * len(dl) // a.grad_accum
    sched = cosine_with_warmup(opt, a.warmup_steps, total)

    loss_fn, miner = ms_loss(acc, a.ms_alpha, a.ms_beta, a.ms_base), ms_miner(acc, a.miner_eps)

    def place_loss(anchor, labels, ref=None):
        # a separate label tensor for the references: pml treats ``ref_labels is labels`` as the
        # same set and would drop the diagonal, i.e. each caption set paired with its own image
        ref_labels = labels.clone() if ref is not None else None
        return loss_fn(anchor, labels, miner(anchor, labels, ref, ref_labels), ref, ref_labels)

    val_kw = dict(image_size=tuple(a.image_size), max_regions=a.max_regions, images=images)
    val_sets = {n: EVAL_SETS[n](**val_kw) for n in a.val_sets} if acc.is_main_process else {}
    if acc.is_main_process:
        log.info("places=%d  processes=%d  batches/epoch=%d  steps=%d  variant=%s",
                 len(dl.dataset), acc.num_processes, len(dl), total, a.variant)

    step, best, meter, t0 = 0, -1.0, Meter(), time.time()
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in dl:
            b = flatten_places(batch)
            with acc.accumulate(model):
                out = model(b["masks"], b["region_valid"], b["cap_emb"], b["cap_valid"], images=b.get("image"))
                if a.variant == "multimodal":
                    loss = place_loss(out.float(), b["label"])
                else:
                    p_img, p_text = out[0].float(), out[1].float()
                    loss = place_loss(p_text, b["label"], p_img) + place_loss(p_img, b["label"], p_text)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if not acc.sync_gradients:
                continue
            sched.step()
            step += 1
            meter.update(loss=loss.item() / acc.num_processes)  # pml's wrapper scales the loss by world size
            if step % a.log_every == 0 and acc.is_main_process:
                log.info("ep %d step %d/%d  %s  lr=%.2e  %.0fs", epoch, step, total, meter, sched.get_last_lr()[0], time.time() - t0)
                meter.reset()
            if step >= total:
                break

        acc.wait_for_everyone()
        state = dict(model=acc.unwrap_model(model).state_dict(), cfg=cfg.to_dict(), args=vars(a), step=step)
        save(acc, Path(a.out) / "last.pt", **state)
        if val_sets:
            recalls = {n: evaluate(acc.unwrap_model(model), ds, acc.device, num_workers=a.num_workers) for n, ds in val_sets.items()}
            for n, r in recalls.items():
                log.info("VAL %-14s epoch %d  %s", n, epoch, format_recalls(r))
            primary = recalls[a.val_sets[0]][1]
            if primary > best:
                best = primary
                save(acc, Path(a.out) / "best.pt", **state, recalls=recalls)
        acc.wait_for_everyone()
        if step >= total:
            break


if __name__ == "__main__":
    main()
