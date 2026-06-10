<div align="center">

# Date Palm Segmentation from WorldView-3 Satellite Data

**Semantic segmentation and individual-tree mapping of date palms from very-high-resolution satellite and aerial imagery.**

[![License: Apache 2.0](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Weights: CC BY 4.0](https://img.shields.io/badge/Weights-CC--BY--4.0-green.svg)](https://huggingface.co/brakuta/date-palm-wv3-models)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-yellow.svg)](https://huggingface.co/brakuta/date-palm-wv3-models)
[![Built on MMSeg](https://img.shields.io/badge/Built%20on-MMSegmentation%201.x-orange.svg)](https://github.com/open-mmlab/mmsegmentation)

</div>

---

A toolkit for detecting, delineating, and counting individual date palm trees
(*Phoenix dactylifera* L.) in WorldView-3 (8-band multispectral) and RGB imagery.
It pairs transformer-based semantic segmentation with an instance-extraction
pipeline that converts a segmentation into georeferenced crown polygons with
per-tree attributes.

Date palm plantations in the United Arab Emirates are threatened by soil
salinity, drought, and the red palm weevil, making accurate, large-scale
inventories important for agriculture, food security, and conservation. The
approach was developed and validated across 275 km² of WorldView-3 scenes over
Ajman, Sharjah, and Fujairah (Al-Ruzouq et al., 2024), demonstrating that
individual palm crowns can be mapped from satellite data despite the resolution
limits that previously confined this task to aerial and UAV imagery.

<div align="center">

| | |
|---|---|
| **14 pretrained models** | five architectures, multispectral and RGB |
| **Three operating modes** | segment · extract trees · end-to-end |
| **Scales to large rasters** | tiled inference, batch folders, country-scale |
| **Reproducible** | pinned input contracts, checkpoint introspection, tests |

</div>

## Table of contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Pretrained models](#pretrained-models)
- [Study area and data](#study-area-and-data)
- [Reported accuracy](#reported-accuracy)
- [Usage](#usage)
- [Data preparation and fine-tuning](#data-preparation-and-fine-tuning)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## What it does

![Example date palm segmentation: 8-band WorldView-3 input, predicted palm mask, and delineated individual crowns](docs/images/prediction_example.png)

*Example output: a WorldView-3 tile, the predicted palm segmentation, and the extracted individual-tree crowns.*


The toolkit exposes three capabilities, each usable on its own:

1. **Semantic segmentation** — run a trained model over a large georeferenced
   raster and obtain a class-label map and an optional palm-probability heatmap.
2. **Individual-tree extraction** — convert a segmentation into delineated,
   counted, georeferenced crowns. Works from a probability heatmap, or from a
   binary mask alone.
3. **End-to-end** — segmentation followed by tree extraction in a single command.

Outputs are written as GeoTIFF (label, heatmap) and GeoPackage (crown polygons
with per-tree score, circularity, and crown diameter in metres).

## Installation

Two parts of the stack are version-sensitive — GDAL (raster I/O) and
PyTorch/CUDA (GPU). The toolkit is designed around both: raster I/O uses
`rasterio` (which bundles its own GDAL, so no separate GDAL install is needed),
and a `palmseg doctor` command verifies the environment and prints the exact
fix for anything missing.

```bash
# 1. environment + raster I/O (rasterio bundles GDAL)
conda create -n palmseg python=3.10 -y && conda activate palmseg
pip install rasterio geopandas shapely scikit-image

# 2. PyTorch matching your CUDA (check 'nvidia-smi', then pick from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. the MM stack via MIM (resolves mmcv/mmengine builds automatically)
pip install -U openmim
mim install mmengine "mmcv>=2.1.0,<2.2.0" "mmsegmentation>=1.2.0"
mim install "mmdet>=3.0.0"          # only for the Mask2Former models

# 4. this toolkit
git clone https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git
cd Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data
pip install -e .

# 5. verify
palmseg doctor
```

Full guide, including a CUDA walkthrough and a PyTorch-free install for
extraction-only use: **[docs/INSTALL.md](docs/INSTALL.md)**.

## Quick start

```bash
# try the full workflow on synthetic data — no GPU, no weights needed
python examples/make_demo_data.py --out-dir demo
palmseg extract --label demo/scene_label.tif --heatmap demo/scene_prob.tif \
                --out demo/scene_trees.gpkg
```

Then, with a real model:

```bash
palmseg download segformer_b5_ms          # fetch weights from Hugging Face
palmseg inspect  weights/segformer_b5_ms.safetensors
palmseg segment-and-extract --model segformer_b5_ms \
        --checkpoint weights/segformer_b5_ms.safetensors \
        --input scene.tif --out out/
```

Step-by-step tutorial: **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

## Pretrained models

All weights are hosted on Hugging Face at
**[brakuta/date-palm-wv3-models](https://huggingface.co/brakuta/date-palm-wv3-models)**
in `.safetensors` format (tensor-only; no executable payload). Download any
model by its ID with `palmseg download <id>`, which fetches the file linked below.

The **Output** column indicates whether a model supports the tree-counting
heatmap (conv-seg heads) or produces masks only (Mask2Former's query head).

### Multispectral (8-band WorldView-3)

| Model ID | Architecture | Output | Size | mIoU (%) | mF-score (%) | Config | Weights |
|---|---|---|---|---|---|---|---|
| `segformer_b3_ms` | SegFormer | heatmap + mask | 179 MB | 76.62 | 84.98 | [`segformer_b3_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/segformer_b3_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/segformer_b3_ms.safetensors) |
| `segformer_b5_ms` | SegFormer | heatmap + mask | 328 MB | — | — | [`segformer_b5_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/segformer_b5_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/segformer_b5_ms.safetensors) |
| `upernet_swin_s_ms` | UPerNet + Swin | heatmap + mask | 326 MB | — | — | [`upernet_swin_s_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/upernet_swin_s_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/upernet_swin_s_ms.safetensors) |
| `upernet_swin_b_ms` | UPerNet + Swin | heatmap + mask | 486 MB | — | — | [`upernet_swin_b_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/upernet_swin_b_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/upernet_swin_b_ms.safetensors) |
| `upernet_vit_deit_s_ms` | UPerNet + ViT-DeiT | heatmap + mask | 234 MB | 71.57 | 80.51 | [`upernet_vit_deit_s_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/upernet_vit_deit_s_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/upernet_vit_deit_s_ms.safetensors) |
| `uniformer_fpn_global_ms` | UniFormer | heatmap + mask | 101 MB | 77.88 | 86.01 | [`uniformer_base_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/uniformer_base_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/uniformer_fpn_global_ms.safetensors) |
| `uniformer_xs_ms` | UniFormer | heatmap + mask | 79 MB | — | — | [`uniformer_xs_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/uniformer_xs_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/uniformer_xs_ms.safetensors) |
| `mask2former_swin_b_ms` | Mask2Former | mask only | 432 MB | — | — | [`mask2former_swin_b_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/mask2former_swin_b_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/mask2former_swin_b_ms.safetensors) |
| `mask2former_swin_s_ms` | Mask2Former | mask only | 275 MB | — | — | [`mask2former_swin_s_ms.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/mask2former_swin_s_ms.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/mask2former_swin_s_ms.safetensors) |

### RGB (3-band)

| Model ID | Architecture | Output | Size | mIoU (%) | mF-score (%) | Config | Weights |
|---|---|---|---|---|---|---|---|
| `segformer_b3_rgb` | SegFormer | heatmap + mask | 179 MB | 76.16 | 84.59 | [`segformer_b3_rgb.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/segformer_b3_rgb.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/segformer_b3_rgb.safetensors) |
| `upernet_swin_t_rgb` | UPerNet + Swin | heatmap + mask | 240 MB | 76.12 | 84.56 | [`upernet_swin_t_rgb.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/upernet_swin_t_rgb.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/upernet_swin_t_rgb.safetensors) |
| `upernet_vit_deit_s_rgb` | UPerNet + ViT-DeiT | heatmap + mask | 232 MB | 75.04 | 83.64 | [`upernet_vit_deit_s_rgb.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/upernet_vit_deit_s_rgb.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/upernet_vit_deit_s_rgb.safetensors) |
| `uniformer_xs_rgb` | UniFormer | heatmap + mask | 79 MB | — | — | [`uniformer_xs_rgb.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/uniformer_xs_rgb.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/uniformer_xs_rgb.safetensors) |
| `mask2former_swin_t_rgb` | Mask2Former | mask only | 190 MB | 76.21 | 84.63 | [`mask2former_swin_t_rgb.py`](https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data/blob/main/configs/mask2former_swin_t_rgb.py) | [`.safetensors`](https://huggingface.co/brakuta/date-palm-wv3-models/blob/main/mask2former_swin_t_rgb.safetensors) |

> **UniFormer models** (`uniformer_*`) use the custom `UniFormer` /
> `UniFormer_Light` backbones, which must be registered in your MMSegmentation
> install (see [docs/MODELS.md](docs/MODELS.md)). The other architectures need
> nothing extra.

```bash
palmseg download --list          # list every model ID
palmseg download --all           # download all released weights
```

## Study area and data

The released models were trained and evaluated on **20 multidate WorldView-3
scenes** covering roughly **275 km²** across Ajman, Sharjah, and the Fujairah
region (Kalba, Dibba, Masafi) in the UAE. WorldView-3 provides eight
multispectral bands at 1.24 m and a panchromatic band at 0.31 m; the
multispectral bands were pansharpened to 0.31 m. Ground truth was created by
manual delineation guided by field campaigns, yielding **16,260 image tiles of
512 × 512 px** for model development (about 1,800 held out for evaluation),
drawn from an analysis of 76,950 individual date palm trees spanning a wide
range of crown sizes, ages, and surrounding environments.

WorldView-3 band layout (the multispectral input contract; see
[docs/PREPROCESSING.md](docs/PREPROCESSING.md)):

| Index | Band | Wavelength (nm) |
|---|---|---|
| 0 | Coastal | 400–450 |
| 1 | Blue | 450–510 |
| 2 | Green | 510–580 |
| 3 | Yellow | 585–625 |
| 4 | Red | 630–690 |
| 5 | Red Edge | 705–745 |
| 6 | NIR-1 | 770–895 |
| 7 | NIR-2 | 860–1040 |

## Reported accuracy

Across architectures, using the **full eight multispectral bands** consistently
outperformed RGB and RGB+NIR composites. In the source study the strongest
multispectral configurations reached roughly **77–78% mIoU and 86% mean
F-score** on the held-out test set, with UniFormer, UPerNet-Swin, and
Mask2Former each improving by about 2% mIoU when moving from RGB to the eight-band
input. On independent scenes from the Dibba region the leading models retained
**83–84% mIoU and 90–91% mean F-score**, indicating good generalisation to new
locations and acquisition dates. Per-model numbers for your own released weights
should be recorded in [docs/MODELS.md](docs/MODELS.md).


Accuracy is reported from Al-Ruzouq et al. (2024), Table 3, on the held-out test
set: the eight-band result for the multispectral models and the RGB result for
the RGB models. A dash (—) marks released size variants that were not separately
evaluated in that table (for example SegFormer-B5, Swin-S/B, and the UniFormer-XS
light variant); run `palmseg` on your own labelled test set to populate these.

## Usage

```bash
# (a) segmentation only -> label raster (+ heatmap)
palmseg segment --model segformer_b5_ms \
        --checkpoint weights/segformer_b5_ms.safetensors \
        --input scene.tif --out out/          # add --no-heatmap for label only

# (b) individual trees from existing rasters (heatmap optional)
palmseg extract --label out/scene_label.tif --heatmap out/scene_prob.tif \
        --out out/scene_trees.gpkg --method watershed --geometry circle

# (c) end-to-end
palmseg segment-and-extract --model segformer_b5_ms \
        --checkpoint weights/segformer_b5_ms.safetensors \
        --input scene.tif --out out/

# batch: point any mode at a folder (resume + per-file fault isolation + CSV)
palmseg segment-and-extract --model segformer_b5_ms \
        --checkpoint weights/segformer_b5_ms.safetensors \
        --input-dir scenes/ --out out/ --summary-csv out/summary.csv
```

Key `extract` options: `--method {watershed,voronoi}`,
`--geometry {circle,polygon}`, `--min-area`, `--min-circularity`,
`--min-seed-distance`, `--dedup-iou`.

## Data preparation and fine-tuning

```bash
# 1. tile mosaics + polygon annotations into the MMSeg layout (edit config block)
python -m palmseg.tools.tile_pipeline

# 2. validate the prepared dataset
palmseg prep-check --data-root data/palm_ms --modality ms --num-classes 2

# 3. (optional) expand a 3-channel ImageNet stem to 8 channels for MS fine-tuning
palmseg adapt-stem mit_b5_ade20k.pth mit_b5_8band.pth --new-in 8

# 4. train with MMSegmentation using a config from configs/
python tools/train.py configs/segformer_b5_ms.py     # from an mmseg clone
```

Full walkthroughs: **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** and
**[docs/FINETUNE.md](docs/FINETUNE.md)**. Fine-tuning supports two or more classes.

## How it works

```
                    (a) segment                     (b) extract
  GeoTIFF scene ─► tiled inference ─► label.tif        existing rasters
                   (seam-blended)     (+ prob.tif)     (label [+ heatmap])
                                           │                  │
                                           └────────┬─────────┘
                                                    ▼
                                     individual-tree extraction
                                       seeds (prob peaks │ distance transform)
                                       assignment (watershed │ voronoi)
                                       delineation (contours + metrics)
                                       georeferenced vectorisation
                                                    ▼
                                       trees.gpkg  (crown polygons)
            (c) segment-and-extract = (a) then (b)
```

The model loader reads input channels, class count, and modality directly from
each checkpoint and reconciles the config to match, so a config's defaults can
never silently disagree with the trained weights.

## Repository layout

| Path | Description |
|---|---|
| [`palmseg/`](palmseg) | the Python package |
| &nbsp;&nbsp;[`cli.py`](palmseg/cli.py) | `palmseg` command-line interface |
| &nbsp;&nbsp;[`doctor.py`](palmseg/doctor.py) | environment diagnostics (GDAL, CUDA, MM stack) |
| &nbsp;&nbsp;[`loader.py`](palmseg/loader.py) | checkpoint-introspecting model loader (`.safetensors` / `.pth`) |
| &nbsp;&nbsp;[`pipeline.py`](palmseg/pipeline.py) | three modes: segment / extract / segment-and-extract |
| &nbsp;&nbsp;[`batch.py`](palmseg/batch.py) | folder-level batch processing (resume, fault isolation) |
| &nbsp;&nbsp;[`inference/tiled.py`](palmseg/inference/tiled.py) | large-raster tiled inference + heatmap export |
| &nbsp;&nbsp;[`postprocess/individual_trees.py`](palmseg/postprocess/individual_trees.py) | seeds + watershed/voronoi + delineation |
| &nbsp;&nbsp;[`postprocess/vectorize.py`](palmseg/postprocess/vectorize.py) | georeferenced vectoriser (+ overlap dedup) |
| &nbsp;&nbsp;[`datasets/palm_dataset.py`](palmseg/datasets/palm_dataset.py) | 2-or-N-class dataset |
| &nbsp;&nbsp;[`transforms/loading.py`](palmseg/transforms/loading.py) | multispectral loader (rasterio-first, GDAL fallback) |
| &nbsp;&nbsp;[`tools/tile_pipeline.py`](palmseg/tools/tile_pipeline.py) | mosaic + polygons → MMSeg tiles (data prep) |
| &nbsp;&nbsp;[`tools/adapt_input_stem.py`](palmseg/tools/adapt_input_stem.py) | 3→N channel stem adapter (fine-tune prep) |
| &nbsp;&nbsp;[`tools/prep_check.py`](palmseg/tools/prep_check.py) | dataset validator |
| &nbsp;&nbsp;[`weights_manifest.py`](palmseg/weights_manifest.py) | model registry + Hugging Face download |
| [`configs/`](configs) | 14 model configs + [`_base_/`](configs/_base_) |
| [`scripts/to_safetensors.py`](scripts/to_safetensors.py) | convert `.pth` checkpoints to `.safetensors` |
| [`scripts/rename_weights.py`](scripts/rename_weights.py) | rename work-dir checkpoints to manifest names |
| [`examples/make_demo_data.py`](examples/make_demo_data.py) | synthetic scene to try the workflow with no GPU |
| [`tests/`](tests) | pytest suite (no GPU or weights needed) |
| [`docs/`](docs) | documentation (see table below) |

## Documentation

| Guide | Contents |
|---|---|
| [INSTALL](docs/INSTALL.md) | environment setup, GDAL and CUDA, troubleshooting |
| [QUICKSTART](docs/QUICKSTART.md) | end-to-end tutorial, including a no-GPU path |
| [MODELS](docs/MODELS.md) | model details, size tiers, custom-backbone notes |
| [PREPROCESSING](docs/PREPROCESSING.md) | the multispectral and RGB input contracts |
| [DATA_PREPARATION](docs/DATA_PREPARATION.md) | tiling mosaics and annotations |
| [FINETUNE](docs/FINETUNE.md) | training on your own labels |

## Citation

This toolkit implements and extends the date palm segmentation method described
in Al-Ruzouq et al. (2024). If you use it, please cite the paper:

```bibtex
@article{alruzouq2024datepalm,
  title   = {Spectral--Spatial transformer-based semantic segmentation for
             large-scale mapping of individual date palm trees using very
             high-resolution satellite data},
  author  = {Al-Ruzouq, Rami and Gibril, Mohamed Barakat A. and Shanableh,
             Abdallah and Bolcek, Jan and Lamghari, Fouad and Hammour,
             Nezar Atalla and El-Keblawy, Ali and Jena, Ratiranjan},
  journal = {Ecological Indicators}, volume = {163}, pages = {112110},
  year    = {2024}, doi = {10.1016/j.ecolind.2024.112110}
}
```

## License

Code is released under the [Apache 2.0](LICENSE) license. Pretrained weights are
released under CC-BY-4.0. See [docs/MODELS.md](docs/MODELS.md) for per-model details.
