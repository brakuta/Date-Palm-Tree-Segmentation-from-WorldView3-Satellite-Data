# Installation

Installation is the part people get stuck on, almost always for two reasons:
**GDAL** (reading geospatial rasters) and **PyTorch/CUDA** (using the GPU). This
guide handles both explicitly, and the toolkit ships a `palmseg doctor` command
that checks your environment and prints the exact fix for anything missing.

Pick your path:

- **Just want to extract trees from rasters you already have?** -> "A. Minimal
  install" (no PyTorch, no GDAL headaches).
- **Want to run the models / fine-tune?** -> "B. Full install".

At any point, run `palmseg doctor` to see what works and what to fix.

---

## A. Minimal install (extraction only, mode b)

Individual-tree extraction needs only a raster backend and a few scientific
libraries — no PyTorch, no MMSegmentation.

```bash
conda create -n palmseg python=3.10 -y
conda activate palmseg
pip install rasterio geopandas shapely scikit-image scipy numpy opencv-python-headless
git clone https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git
cd Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data
pip install -e .
palmseg doctor          # "extraction-only (mode b): YES"
```

`rasterio` bundles its own GDAL, so you do **not** need to install GDAL
separately. This is the recommended way to get geospatial I/O working.

---

## B. Full install (run models, fine-tune)

### Step 1 — environment + raster backend (the GDAL fix)

```bash
conda create -n palmseg python=3.10 -y
conda activate palmseg
pip install rasterio geopandas shapely scikit-image
python -c "import rasterio; print('rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__)"
```

Why this works: historically people `conda install gdal` or `pip install GDAL`
and fight version/ABI errors. **rasterio ships a working GDAL inside its wheel**,
so installing rasterio is enough for everything this toolkit reads/writes. The
custom multispectral loader uses rasterio first and only falls back to a
standalone `osgeo.gdal` if rasterio is absent.

> If you specifically need the `osgeo` package for other tools, install it from
> conda-forge into the same env: `conda install -c conda-forge gdal=3.6`. It is
> optional here.

### Step 2 — find your CUDA version, then install the matching PyTorch

The #1 GPU mistake is installing a PyTorch build whose CUDA version does not
match the machine. Do this:

1. Run `nvidia-smi`. Top-right shows **"CUDA Version: XX.X"** — this is the
   maximum CUDA your driver supports.
2. Go to https://pytorch.org/get-started/locally/ and pick a CUDA <= that
   number. Common, well-supported choices:

   ```bash
   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # CPU only (no GPU; inference works but is slow)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
3. Verify the GPU is actually visible:

   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

   If this prints `False` on a GPU machine, you installed a CPU-only or
   mismatched build. `palmseg doctor` will say exactly that and give the fix.

### Step 3 — the MM stack via MIM

MIM resolves compatible `mmcv`/`mmengine` builds for your PyTorch+CUDA, which is
more reliable than installing them by hand.

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.1.0,<2.2.0"
mim install "mmsegmentation>=1.2.0"
mim install "mmdet>=3.0.0"        # ONLY needed for the Mask2Former models
```

### Step 4 — this toolkit

```bash
git clone https://github.com/brakuta/Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data.git
cd Date-Palm-Tree-Segmentation-from-WorldView3-Satellite-Data
pip install -e .
```

### Step 5 — verify everything

```bash
palmseg doctor
```

You want the summary to read:
```
extraction-only (mode b): YES
segmentation/inference (modes a, c): YES
```

### Step 6 — UniFormer models only

UniFormer is not part of stock MMSegmentation. To use the `uniformer_*` models,
register the UniFormer backbone in your MMSeg install (copy the backbone module
into `mmseg/models/backbones/` and add it to that package's `__init__.py`), or
install the project's MMSeg fork at commit `b040e147`. SegFormer, Swin, ViT, and
Mask2Former need nothing extra.

---

## Troubleshooting (what `palmseg doctor` tells you)

| Symptom | Cause | Fix |
|--------|-------|-----|
| `raster backend: FAIL` | no rasterio/GDAL | `pip install rasterio` |
| `import rasterio` ABI error | broken mixed conda/pip GDAL | `pip uninstall -y gdal rasterio && pip install rasterio` |
| `CUDA available: False` on a GPU box | CPU-only or mismatched torch | reinstall torch with the right `--index-url` (Step 2) |
| `mim install mmcv` finds no wheel | unusual torch/CUDA combo | use CUDA 11.8 or 12.1 torch builds, which have prebuilt mmcv |
| `UniFormer is not in the registry` | backbone not registered | see Step 6 |
| `mmdet is not installed` (Mask2Former) | mmdet missing | `mim install "mmdet>=3.0.0"` |
| `KeyError: PalmDataset / LoadSingleRSImageFromFile` | toolkit not importable in training | `pip install -e .`; configs auto-register via `custom_imports` |

## Relationship to a cloned MMSegmentation (no file overlap)

This toolkit is a **separate package** (`palmseg`). It does not require you to
copy files into, or modify, an mmsegmentation clone, and it does not override
any built-in MMSeg class. Custom modules live in `palmseg/` and register
themselves via each config's `custom_imports`, so MMSeg's own
`tools/train.py <config>` works without mixing files. A clone is only needed for
MMSeg's training script or to register the UniFormer backbone.
