"""Prepare ScanNet frames, instance masks and positive pairs.

    python -m data_prep.scannet --scans_dir datasets/ScanNet/scans \
        --split datasets/ScanNet/scannetv2_train.txt --out_dir datasets/ScanNet

Expects each scan exported with ScanNet's ``SensorData.py`` (``color/``, ``depth/``,
``pose/``, ``intrinsic/``) and its ``<scene>_2d-instance-filt.zip`` unpacked to
``instance-filt/``. For every frame with an instance map it writes

    Images/train/<scene>/<frame>.jpg      the colour frame (a symlink when it already has the
                                          instance map's size, else resized to it)
    Segments/train/<scene>/<frame>.npz    one boolean mask per instance (same format as data_prep.segment)
    Segments/train/<scene>/<frame>.json   XYWH boxes and areas

and one ``Pairs/<scene>.npz`` with the camera centres and the positive pairs: frames of
the same scene between ``--r_min`` and ``--r_pos`` metres apart whose views overlap by
at least ``--min_overlap``. A point of one view counts as overlap when it projects into
the other view and agrees with that view's depth, so occluded points do not count.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

LOOKUP_STRIDE = 4  # depth maps are kept at 1/4 resolution for the visibility test


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(4, 4)


def instance_masks(inst: np.ndarray):
    ids = [i for i in np.unique(inst) if i != 0]
    masks = np.stack([inst == i for i in ids]) if ids else np.zeros((0, *inst.shape), bool)
    bboxes, areas = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        bboxes.append([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
        areas.append(int(m.sum()))
    return masks, bboxes, areas


def backproject(depth: np.ndarray, K: np.ndarray, pose: np.ndarray, stride: int) -> np.ndarray:
    """Sampled depth pixels of one frame as world points (P x 3)."""
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[v, u].astype(np.float64)
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    return (pose @ cam)[:3].T


def load_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32) / 1000.0  # millimetres -> metres


def overlap(points: np.ndarray, K: np.ndarray, pose: np.ndarray, depth: np.ndarray, hw, tol: float) -> float:
    """Fraction of world points visible in the frame with ``pose``: inside its image and
    within ``tol`` (relative) of its depth map ``depth`` (sampled every LOOKUP_STRIDE pixels)."""
    if len(points) == 0:
        return 0.0
    cam = np.linalg.inv(pose) @ np.c_[points, np.ones(len(points))].T
    z = cam[2]
    u = K[0, 0] * cam[0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * cam[1] / np.maximum(z, 1e-6) + K[1, 2]
    inside = (z > 0) & (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    d = np.zeros_like(z)
    d[inside] = depth[v[inside].astype(int) // LOOKUP_STRIDE, u[inside].astype(int) // LOOKUP_STRIDE]
    visible = inside & (d > 0) & (np.abs(z - d) <= tol * d)
    return float(visible.mean())


def process_scene(scene_dir: Path, out_dir: Path, a):
    scene = scene_dir.name
    K = np.loadtxt(scene_dir / "intrinsic" / "intrinsic_depth.txt")[:3, :3]
    frames, poses = [], []
    for inst_path in sorted(scene_dir.glob("instance-filt/*.png"), key=lambda p: int(p.stem)):
        pose = load_pose(scene_dir / "pose" / f"{inst_path.stem}.txt")
        if not np.isfinite(pose).all():  # ScanNet marks failed tracking with -inf poses
            continue
        frames.append(int(inst_path.stem))
        poses.append(pose)

        img_out = out_dir / "Images" / "train" / scene / f"{inst_path.stem}.jpg"
        seg_out = out_dir / "Segments" / "train" / scene / f"{inst_path.stem}.npz"
        if not seg_out.exists():
            img_out.parent.mkdir(parents=True, exist_ok=True)
            seg_out.parent.mkdir(parents=True, exist_ok=True)
            inst = np.array(Image.open(inst_path))
            colour_path = (scene_dir / "color" / f"{inst_path.stem}.jpg").resolve()
            colour = Image.open(colour_path)
            img_out.unlink(missing_ok=True)
            if colour.size == inst.shape[::-1]:
                img_out.symlink_to(colour_path)
            else:  # DAM needs the image and its masks at the same size
                colour.convert("RGB").resize(inst.shape[::-1], Image.BILINEAR).save(img_out)
            masks, bboxes, areas = instance_masks(inst)
            np.savez_compressed(seg_out, masks=masks)
            with open(seg_out.with_suffix(".json"), "w") as f:
                json.dump({"bboxes": bboxes, "areas": areas}, f)

    if not frames:
        return
    poses = np.stack(poses)
    centres = poses[:, :3, 3]
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    depth_path = lambda f: scene_dir / "depth" / f"{f}.png"
    hw = load_depth(depth_path(frames[0])).shape
    small = [load_depth(depth_path(f))[::LOOKUP_STRIDE, ::LOOKUP_STRIDE] for f in frames]
    pairs = []
    for i in range(len(frames)):
        cand = np.flatnonzero((dist[i] >= a.r_min) & (dist[i] <= a.r_pos))
        cand = cand[cand > i]
        if len(cand) == 0:
            continue
        pts = backproject(load_depth(depth_path(frames[i])), K, poses[i], a.stride)
        pairs += [(i, j) for j in cand if overlap(pts, K, poses[j], small[j], hw, a.depth_tol) >= a.min_overlap]

    (out_dir / "Pairs").mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "Pairs" / f"{scene}.npz", frames=np.array(frames), centres=centres,
             pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans_dir", required=True)
    ap.add_argument("--split", required=True, help="scannetv2_train.txt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--r_min", type=float, default=0.3)
    ap.add_argument("--r_pos", type=float, default=1.0)
    ap.add_argument("--min_overlap", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=16, help="pixel stride when sampling depth for the overlap test")
    ap.add_argument("--depth_tol", type=float, default=0.1, help="relative depth agreement for a point to count as visible")
    a = ap.parse_args()

    scenes = Path(a.split).read_text().split()
    for scene in tqdm(scenes, desc="ScanNet scenes"):
        process_scene(Path(a.scans_dir) / scene, Path(a.out_dir), a)


if __name__ == "__main__":
    main()
