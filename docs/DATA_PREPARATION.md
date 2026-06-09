# Data preparation

To fine-tune (or retrain) you need image tiles and matching label-mask tiles in
the MMSegmentation folder layout. This toolkit ships a production tiler,
`palmseg/tools/tile_pipeline.py`, that turns georeferenced **mosaics + polygon
annotations** into that layout, with train/val/test splits, class encoding, and
audit logs.

You only edit a configuration block at the top of the file and a `JOBS` list,
then run it. No command-line arguments.

## What you provide

1. A **mosaic** (orthomosaic / satellite scene) as a GeoTIFF.
2. **Polygon annotations** of your classes (shapefile or geodatabase layer).
3. Optionally, a **split layer**: a polygon shapefile whose features mark which
   regions are train / val / test (recommended so splits are spatially disjoint
   and there is no train/test leakage).

## The output layout (what MMSeg expects)

```
OUTPUT_DIR/
  img_dir/
    train/   *.tif      image tiles
    val/     *.tif
    test/    *.tif
  ann_dir/
    train/   *.png      single-band class-index masks (0=bg, 1..K=classes)
    val/     *.png
    test/    *.png
  class_palette.json    the class -> index mapping used
```

This is exactly what the dataset configs in `configs/_base_/` point to via
`data_root`.

## Configure the tiler

Open `palmseg/tools/tile_pipeline.py` and set the constants near the top:

| Setting | Meaning | Typical |
|---------|---------|---------|
| `OUTPUT_DIR` | where tiles are written | your path |
| `TILE_SIZE` | tile side in pixels | `512` (match the model crop) |
| `OVERLAP` | training-tile overlap in px (both axes) | `64`–`128` |
| `CLASS_MAP` | class name -> index (background is always 0) | `{"DatePalm": 1}` |
| `APPLY_CONTRAST_STRETCH` | per-band p2/p98 stretch to 8-bit | see contract below |
| `BANDS` | which bands to keep | `None` (all) or `[1,2,3]` (RGB from RGBA) |
| `KEEP_BACKGROUND_TILES` | keep tiles with no annotation | usually `False` |
| `DROP_NODATA_TILES` | skip all-black/nodata tiles | `True` |
| `MIN_IMAGE_COVERAGE` | min non-zero pixel fraction to keep a tile | `0.5` |

### The contrast-stretch contract (read this)

`APPLY_CONTRAST_STRETCH` interacts with the model modality and must match it:

- **Multispectral (8-band) models:** set `APPLY_CONTRAST_STRETCH = False`. The
  released MS models were trained on **raw 0-255 DN with no stretch**
  (preprocessor mean=0/std=1). A stretched export changes the input
  distribution and degrades transfer.
- **RGB models:** `APPLY_CONTRAST_STRETCH = True` is fine; the RGB path
  normalises with ImageNet statistics regardless.

See `docs/PREPROCESSING.md` for the full input contract.

## Define the jobs

`JOBS` is a list of dicts, one per mosaic. Minimal example (single class, single
shapefile, split by a region layer):

```python
JOBS = [
    {
        "name":   "WV3_Ajman",
        "mosaic": Path(r"/data/Ajman_WV3_2021_30cm.tif"),
        "sources": [
            {"shapefile": Path(r"/data/Datepalm.gdb"),
             "layer": "WV3_Datepalm_Ajman",
             "class_name": "DatePalm"},
        ],
        "split_shapefile": Path(r"/data/Datepalm_WV3_Area.shp"),
        "split_field":     "Task",     # feature attribute holding train/val/test
        "overlap":         64,
        "bands":           [1, 2, 3],  # or None to keep all 8 MS bands
    },
]
```

Variations the tiler supports:
- **Multiple classes** in separate layers: add more entries to `sources`, each
  with its own `class_name` (and set `CLASS_MAP` with all of them).
- **One shapefile, multiple classes** in a column: use `"class_field": "species"`
  in the source; the column values must be keys in `CLASS_MAP`.
- **Simple split**: replace `split_shapefile`/`split_field` with `"split": "train"`
  to send the whole job to one split.

## Run it

```bash
python -m palmseg.tools.tile_pipeline
```

It prints per-job progress and writes an audit log. For 8-band MS data, confirm
`APPLY_CONTRAST_STRETCH = False` first.

## Validate before training

Always check the result:

```bash
palmseg prep-check --data-root OUTPUT_DIR --modality ms --num-classes 2
```

This verifies image/mask pairing, single-band index masks, that mask values are
within `[0, num_classes-1]` (255 allowed as ignore), and that the image band
count matches the modality. Fix any reported issue before fine-tuning.

Next: `docs/FINETUNE.md`.
