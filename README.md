# Date Palm Segmentation from WorldView-3 Satellite Data

A toolkit for mapping date palm trees from very-high-resolution satellite and
aerial imagery, using transformer-based semantic segmentation (MMSegmentation)
and an individual-tree extraction pipeline.

It does three things, each runnable on its own:

- **(a) Semantic segmentation** — run a trained model over a large georeferenced
  raster -> class-label map (and optional palm-probability heatmap).
- **(b) Individual-tree extraction** — turn a segmentation into delineated,
  counted, georeferenced crowns. Works from a probability heatmap **or a binary
  mask alone**.
- **(c) End-to-end** — segmentation then tree extraction in one call.

Supports **multispectral 8-band WorldView-3** and **RGB** imagery, five
architectures (SegFormer, UPerNet+Swin, UPerNet+ViT-DeiT, UniFormer,
Mask2Former), single-image or **whole-folder batch** processing, and
fine-tuning on your own data (2 or more classes).

> Code: Apache-2.0. Weights: CC-BY-4.0. Built on MMSegmentation 1.x.

---

## Start here

| I want to… | Go to |
|------------|-------|
| Install (GDAL & CUDA setup) | [`docs/INSTALL.md`](docs/INSTALL.md) |
| Check my environment | run `palmseg doctor` |
| Learn the workflow end to end | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| Prepare training data | [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) |
| Fine-tune on my own labels | [`docs/FINETUNE.md`](docs/FINETUNE.md) |
| Understand the input contract | [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md) |
| See the models & accuracy | [`docs/MODELS.md`](docs/MODELS.md) |
| Publish to GitHub (step by step) | [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md) |
| Create HF account and upload weights | [`docs/HUGGINGFACE_WEIGHTS.md`](docs/HUGGINGFACE_WEIGHTS.md) |

---

## Install (short version)

Two common setup issues and how they are handled:

- **GDAL.** Don't install GDAL by hand. Install **rasterio**, which bundles its
  own GDAL; the toolkit's raster loader uses it automatically. `pip install rasterio`.
- **PyTorch/CUDA.** Run `nvidia-smi`, read the CUDA version top-right, install
  the matching PyTorch from pytorch.org. `palmseg doctor` tells you if you got
  it wrong and prints the exact command.

```bash
conda create -n palmseg python=3.10 -y && conda activate palmseg
pip install rasterio geopandas shapely scikit-image
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
pip install -U openmim && mim install mmengine "mmcv>=2.1.0,<2.2.0" "mmsegmentation>=1.2.0"
mim install "mmdet>=3.0.0"     # only for Mask2Former models
git clone https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git
cd Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data && pip install -e .
palmseg doctor                 # verify GDAL + CUDA + MM stack
```

For extraction only (mode b) you don't need PyTorch or MMSegmentation — see the
minimal install in [`docs/INSTALL.md`](docs/INSTALL.md).

---

## Inferencing

```bash
# get a model (set HF_REPO_ID in palmseg/weights_manifest.py first)
palmseg download segformer_b5_ms
palmseg inspect  weights/segformer_b5_ms.pth     # channels / classes / modality

# (a) segmentation: one scene -> label (+ heatmap)
palmseg segment --model segformer_b5_ms --checkpoint weights/segformer_b5_ms.pth \
                --input scene.tif --out out/

# (b) individual trees from existing rasters (heatmap optional)
palmseg extract --label out/scene_label.tif --heatmap out/scene_prob.tif \
                --out out/scene_trees.gpkg

# (c) end to end
palmseg segment-and-extract --model segformer_b5_ms \
                --checkpoint weights/segformer_b5_ms.pth --input scene.tif --out out/

# batch: point any mode at a folder (resume + fault isolation + CSV summary)
palmseg segment-and-extract --model segformer_b5_ms \
                --checkpoint weights/segformer_b5_ms.pth \
                --input-dir scenes/ --out out/ --summary-csv out/summary.csv
```

Output `trees.gpkg` carries per-crown `score`, `circularity`, and `diam_m`.

---

## Data preparation & fine-tuning

```bash
# 1. tile mosaics + polygon annotations into MMSeg layout (edit the config block)
python -m palmseg.tools.tile_pipeline

# 2. validate the prepared dataset
palmseg prep-check --data-root data/palm_ms --modality ms --num-classes 2

# 3. (optional) expand a 3-channel ImageNet stem to 8 channels for MS fine-tuning
palmseg adapt-stem mit_b5_ade20k.pth mit_b5_8band.pth --new-in 8

# 4. train with MMSegmentation using a config from configs/
python tools/train.py configs/finetune_palm_ms.py   # from an mmseg clone
```

Full walkthroughs: [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) and
[`docs/FINETUNE.md`](docs/FINETUNE.md). Fine-tuning supports 2 or more classes.

---

## How it works

```
                    (a) segment                     (b) extract
  GeoTIFF scene ─► tiled inference ─► label.tif        existing rasters
                   (seam-blended)     (+prob.tif)      (label [+heatmap])
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

## Repository layout

```
palmseg/
  cli.py                    `palmseg` command-line interface
  doctor.py                 environment diagnostics (GDAL, CUDA, MM stack)
  loader.py                 checkpoint-introspecting model loader
  pipeline.py               three modes: segment / extract / segment_and_extract
  batch.py                  folder-level batch processing (resume, fault isolation)
  inference/tiled.py        large-raster tiled inference + heatmap export
  postprocess/
    individual_trees.py     seeds + watershed/voronoi + delineation (heatmap opt.)
    vectorize.py            georeferenced vectoriser (+ overlap dedup)
  datasets/palm_dataset.py  2-or-N-class dataset
  transforms/loading.py     multispectral loader (rasterio-first, GDAL fallback)
  tools/
    tile_pipeline.py        mosaic+polygons -> MMSeg tiles (data prep)
    adapt_input_stem.py     3->N channel stem adapter (fine-tune prep)
    prep_check.py           dataset validator
  weights_manifest.py       model registry + Hugging Face download
configs/                    5 architectures x {ms, rgb} + _base_
examples/make_demo_data.py  synthetic scene to try the workflow with no GPU
tests/                      pytest suite (no GPU/weights needed)
docs/                       INSTALL, QUICKSTART, DATA_PREPARATION, FINETUNE,
                            PREPROCESSING, MODELS, GITHUB_SETUP
```

## Method reference and citation

This toolkit implements and extends the date palm segmentation method described
in Al-Ruzouq et al. (2024), *Ecological Indicators* 163, 112110. The codebase
(model loading, tiled inference, the three-mode pipeline, individual-tree
extraction, batch processing, and tooling) is maintained as a standalone
project.

Cite the paper for the segmentation method and the trained models:

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

Cite the software (see `CITATION.cff`):

```bibtex
@software{gibril_datepalm_wv3,
  title  = {Date Palm Segmentation from WorldView-3 Satellite Data},
  author = {Gibril, Mohamed Barakat A.},
  year   = {2024},
  url    = {https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data}
}
```
