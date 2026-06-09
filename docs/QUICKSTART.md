# Quickstart tutorial

This walks through every capability end to end. Steps 1–2 need **no GPU, no
model weights** — they teach the workflow on synthetic data. Steps 3–5 use a
real model.

## Step 0 — check your environment

```bash
palmseg doctor
```

This reports whether your raster backend (GDAL), PyTorch/CUDA, and the MM stack
are correctly installed, and prints the exact fix for anything missing. If the
summary shows `extraction-only (mode b): YES` you can do Steps 1-2 immediately.

## Step 1 — generate a demo scene

After installing (see `docs/INSTALL.md`):

```bash
python examples/make_demo_data.py --out-dir demo
```

This writes `demo/scene_rgb.tif` (an RGB scene), `demo/scene_label.tif` (a
class-label raster), and `demo/scene_prob.tif` (a palm-probability heatmap).

## Step 2 — extract individual trees (mode b, no model needed)

Turn the label + heatmap into counted, georeferenced crowns:

```bash
palmseg extract \
  --label   demo/scene_label.tif \
  --heatmap demo/scene_prob.tif \
  --out     demo/scene_trees.gpkg \
  --method  watershed \
  --geometry circle \
  --min-area 30 \
  --min-seed-distance 8
```

Open `demo/scene_trees.gpkg` in QGIS (or `geopandas.read_file`) to see one
polygon per tree, each carrying `score`, `circularity`, and `diam_m`.

Try it **without** a heatmap (works on a binary mask alone):

```bash
palmseg extract --label demo/scene_label.tif --out demo/trees_maskonly.gpkg \
                --min-area 30
```

## Step 3 — get a real model

Set `HF_REPO_ID` in `palmseg/weights_manifest.py` (once weights are uploaded),
then:

```bash
palmseg download --list                 # see available model ids
palmseg download segformer_b5_ms        # downloads to weights/
palmseg inspect weights/segformer_b5_ms.pth
palmseg selftest --model segformer_b5_ms --checkpoint weights/segformer_b5_ms.pth
```

`inspect` should report `in_channels: 8`, `num_classes: 2`, `modality: ms`.

## Step 4 — run segmentation (mode a)

On your own georeferenced scene (8-band for `_ms` models, 3-band for `_rgb`):

```bash
palmseg segment \
  --model      segformer_b5_ms \
  --checkpoint weights/segformer_b5_ms.pth \
  --input      my_scene.tif \
  --out        out/
# writes out/my_scene_label.tif and out/my_scene_prob.tif
# add --no-heatmap to write only the label raster
```

## Step 5 — end to end (mode c)

```bash
palmseg segment-and-extract \
  --model      segformer_b5_ms \
  --checkpoint weights/segformer_b5_ms.pth \
  --input      my_scene.tif \
  --out        out/
# writes label, heatmap, and out/my_scene_trees.gpkg
```


## Batch processing (a whole folder)

Every mode accepts `--input-dir` to process all rasters in a folder, with
per-file fault isolation, resume (skip files already done), and an optional CSV
summary.

```bash
# (a) segment every raster in a folder
palmseg segment --model segformer_b5_ms --checkpoint weights/segformer_b5_ms.pth \
                --input-dir scenes/ --out out/ --summary-csv out/segment_summary.csv

# (b) extract trees for every label raster in a folder (heatmaps matched by name:
#     <stem>_label.tif pairs with <stem>_prob.tif in --heatmap-dir)
palmseg extract --input-dir out/ --heatmap-dir out/ --pattern "*_label.tif" \
                --out trees/ --summary-csv trees/extract_summary.csv

# (c) end-to-end over a folder
palmseg segment-and-extract --model segformer_b5_ms \
                --checkpoint weights/segformer_b5_ms.pth \
                --input-dir scenes/ --out out/ --summary-csv out/batch_summary.csv
```

Re-running resumes by default (already-produced outputs are skipped); pass
`--no-resume` to force reprocessing.

## Fine-tuning (optional)

To train on your own labels (2 or more classes), see `docs/DATA_PREPARATION.md` then `docs/FINETUNE.md`. The
short version:

```bash
# 1. tile your mosaics + polygon annotations into MMSeg-style data
python -m palmseg.tools.tile_pipeline          # edit its CONFIG block first

# 2. validate the prepared dataset
palmseg prep-check --data-root data/palm_ms --modality ms --num-classes 2

# 3. (optional) expand a 3-channel ImageNet stem to 8 channels
palmseg adapt-stem mit_b5_ade20k.pth mit_b5_8band.pth --new-in 8

# 4. train with MMSegmentation's trainer using a config from configs/
python -m mmengine.tools.train configs/finetune_palm_ms.py
```

## Common parameters for `extract`

| Flag | Meaning | Typical |
|------|---------|---------|
| `--method` | `watershed` (default) or `voronoi` | watershed |
| `--geometry` | `circle` (area-preserving) or `polygon` | circle |
| `--prob-thr` | probability cut before peak finding | 0.5 |
| `--min-area` | drop instances smaller than N px | 9–30 |
| `--min-circularity` | reject non-round blobs (0–1) | 0–0.5 |
| `--min-seed-distance` | minimum px between tree centres | 0–10 |
| `--dedup-iou` | merge overlapping crowns above this IoU | 0 (off) |
