# vpr_xmodal

Visual place recognition with segment-level vision-language grounding. Each image is split
into segments, each segment is captioned, and the segment features are grounded in their
captions before being aggregated into one global descriptor. The code covers multimodal
retrieval, where queries and database images are both described by image and captions, and
cross-modal retrieval, where caption-only queries are matched against an image database.

## Installation

```bash
pixi install
```

Download the MobileSAM checkpoint `mobile_sam.pt` from the MobileSAM repository. Captioning
with DAM needs a GPU. Run the commands below from this folder inside `pixi shell`.

## Datasets

Place each dataset under `VPR_DATA_ROOT` (default `./datasets`):

| Dataset | Images | Ground truth |
|---|---|---|
| GSVCities | `GSVCities/Images/<city>/` | `GSVCities/Dataframes/<city>.csv` |
| ScanNet | exported scans, see below | camera poses |
| Pittsburgh | `Pittsburgh/Images/{database,query}/` | `Pittsburgh/pitts30k_{val,test}.mat` (NetVLAD splits) |
| MSLS | `MSLS/train_val/` | `MSLS/msls_val_{dbImages,qImages,qIdx,pIdx}.npy` |
| Baidu | `Baidu/{training,query}_images_undistort/` | `Baidu/{training,query}_gt/*.camera` |
| AmsterTime | `AmsterTime/Images/{database,query}/` | pairs share a file name |

The Pittsburgh folders need the images of both `pitts30k_val`, which is used for validation,
and `pitts30k_test`.

## Data preparation

Segment, caption, clean and embed every split. For one GSV-Cities city:

```bash
D=datasets/GSVCities
python -m data_prep.segment --img_dir $D/Images/Medellin --out_dir $D/Segments/Medellin --checkpoint mobile_sam.pt
python -m data_prep.caption --img_dir $D/Images/Medellin --masks_dir $D/Segments/Medellin --captions_dir $D/Captions/Medellin
python -m data_prep.build_index --captions_dir $D/Captions/Medellin --output $D/RegionIndex/Medellin.csv
python -m data_prep.embed_captions --csv $D/RegionIndex/Medellin.csv --output $D/siglip2_text_embeddings/Medellin.npy
```

Splits are named after the city for GSV-Cities, `database` and `query` for Pittsburgh, Baidu
and AmsterTime, and `val` for MSLS. Baidu and MSLS keep their own image folders, so pass those
as `--img_dir` (for example `datasets/MSLS/train_val` for `val`).

ScanNet uses its own instance masks instead of `data_prep.segment`. Export each training scan
with ScanNet's `SensorData.py`, unzip `<scene>_2d-instance-filt.zip` into the scan folder as
`instance-filt/`, and run the command below. The image folder it writes holds links to the
exported frames, so keep the export in place. Then caption, clean and embed
`datasets/ScanNet/{Images,Segments}/train` as above.

```bash
python -m data_prep.scannet --scans_dir datasets/ScanNet/scans --split scannetv2_train.txt --out_dir datasets/ScanNet
```

Cluster the training captions into pseudo-labels, then build the region caches for every
split and the DINOv2 feature caches for the training sets:

```bash
python -m data_prep.cluster_captions \
    --index_dirs datasets/GSVCities/RegionIndex datasets/ScanNet/RegionIndex \
    --emb_dirs datasets/GSVCities/siglip2_text_embeddings datasets/ScanNet/siglip2_text_embeddings \
    --out datasets/clusters.npz --gpu
python -m vpr_xmodal.scripts.build_region_cache --dataset GSVCities --splits all
python -m vpr_xmodal.scripts.build_region_cache --dataset Pittsburgh --splits database query   # and so on for every dataset
python -m vpr_xmodal.scripts.cache_features --img_dir datasets/GSVCities/Images/Medellin --out datasets/GSVCities/dinov2_cache/Medellin.npy
python -m vpr_xmodal.scripts.cache_features --img_dir datasets/ScanNet/Images/train --out datasets/ScanNet/dinov2_cache/train.npy
```

## Training

Training has two stages: segment-caption grounding, then place recognition. Use
`accelerate launch --multi_gpu --num_processes N` to train on several GPUs.

```bash
accelerate launch -m vpr_xmodal.scripts.train_grounding --variant multimodal --cities all --scannet --cached --out runs/grounding_mm
accelerate launch -m vpr_xmodal.scripts.train_vpr --variant multimodal --cities all --scannet \
    --grounding_ckpt runs/grounding_mm/grounder.pt --out runs/vpr_mm

accelerate launch -m vpr_xmodal.scripts.train_grounding --variant crossmodal --cities all --cached --out runs/grounding_xm
accelerate launch -m vpr_xmodal.scripts.train_vpr --variant crossmodal --cities all \
    --grounding_ckpt runs/grounding_xm/grounder.pt --out runs/vpr_xm
```

## Evaluation

```bash
python -m vpr_xmodal.scripts.evaluate --ckpt runs/vpr_mm/best.pt --datasets pitts30k_test msls_sf msls_cph baidu amstertime
python -m vpr_xmodal.scripts.evaluate --ckpt runs/vpr_xm/best.pt --datasets amstertime msls_val pitts30k_test
```

A query counts as correct when a retrieved image lies within 25 m (Pitts30K, MSLS) or 10 m
(Baidu Mall) of it, or is its pair (AmsterTime).

## Ablations

`train_grounding`:
- `--seg_img mlp` replaces Segment x Image with an MLP.
- `--mining {random,coarse,hierarchical,full}` selects the mining strategy.

`train_vpr`:
- `--modality_dropout 0` turns off modality dropout.
- `--num_clusters {16,32,64,128}` sets the descriptor size.
- `--streams visual` or `--streams text` trains without the text or image stream; omit `--grounding_ckpt`.

Data preparation:
- `build_index --drop_terms signs text scene-text` removes scene text; rerun the steps after it.
- `segment --whole_image` gives whole-image captions. Write the outputs into `*_whole` folders, set
  `VPR_ANNOTATION_SET=whole` for the later steps, and omit `--grounding_ckpt`.
