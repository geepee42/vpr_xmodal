"""Evaluate a trained checkpoint.

    python -m vpr_xmodal.scripts.evaluate --ckpt runs/vpr_mm/best.pt --datasets pitts30k_test msls_sf msls_cph baidu amstertime
    python -m vpr_xmodal.scripts.evaluate --ckpt runs/vpr_xm/best.pt --datasets amstertime msls_val pitts30k_test

The mode follows the checkpoint. Multimodal: database and queries are both described by
image + captions. Cross-modal: database descriptors come from image regions only and
query descriptors from captions only.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.amstertime import amstertime
from ..data.baidu import baidu
from ..data.common import collate_samples
from ..data.msls import msls
from ..data.pitts import pitts
from ..eval_utils import format_recalls, recall_at_k
from ..models import VPRConfig, build_model

EVAL_SETS = {
    "pitts30k_val": lambda **kw: pitts("pitts30k_val", **kw),
    "pitts30k_test": lambda **kw: pitts("pitts30k_test", **kw),
    "msls_val": msls,
    "msls_sf": lambda **kw: msls("sf", **kw),
    "msls_cph": lambda **kw: msls("cph", **kw),
    "baidu": baidu,
    "amstertime": amstertime,
}
KS = (1, 5, 10)


@torch.inference_mode()
def extract_descriptors(model, ds, device, batch_size: int, num_workers: int) -> torch.Tensor:
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_samples)
    crossmodal = model.cfg.variant == "crossmodal"
    out = torch.zeros(len(ds), model.out_dim)
    for b in dl:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        idx = b["index"]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if not crossmodal:
                out[idx.cpu()] = model(b["masks"], b["region_valid"], b["cap_emb"], b["cap_valid"], images=b.get("image")).float().cpu()
                continue
            db, q = idx < ds.num_db, idx >= ds.num_db
            if db.any():
                out[idx[db].cpu()] = model.encode_image(b["image"][db], b["masks"][db], b["region_valid"][db]).float().cpu()
            if q.any():
                out[idx[q].cpu()] = model.encode_text(b["cap_emb"][q], b["cap_valid"][q]).float().cpu()
    return out


def evaluate(model, ds, device, batch_size: int = 64, num_workers: int = 8, ks=KS):
    was_training = model.training
    model.eval()
    d = extract_descriptors(model, ds, device, batch_size, num_workers)
    model.train(was_training)
    return recall_at_k(d[: ds.num_db], d[ds.num_db :], ds.positives, ks=ks)


def load_model(ckpt_path: str, device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = VPRConfig(**{**ck["cfg"], "pretrained": False})  # weights come from the checkpoint
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+", default=[n for n in EVAL_SETS if n != "pitts30k_val"], choices=list(EVAL_SETS))
    ap.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--max_regions", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_json", default="", help="append results to this json file")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, device)
    results = {}
    for name in a.datasets:
        ds = EVAL_SETS[name](image_size=tuple(a.image_size), max_regions=a.max_regions, images=model.cfg.streams != "text")
        results[name] = evaluate(model, ds, device, a.batch_size, a.num_workers)
        print(f"{name:14s} ({ds.num_db:5d} db / {ds.num_q:5d} q)  {format_recalls(results[name])}")

    if a.out_json:
        p = Path(a.out_json)
        history = json.loads(p.read_text()) if p.exists() else []
        history.append({"ckpt": a.ckpt, "variant": model.cfg.variant, "results": results})
        p.write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
