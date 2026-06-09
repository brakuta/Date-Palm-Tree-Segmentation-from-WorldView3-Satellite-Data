# Preprocessing & input contract

Input handling differs by modality, and the two contracts are not
interchangeable. Using the wrong one degrades or breaks results. The toolkit's loader (`palmseg.loader`) enforces the correct one
automatically by inspecting the checkpoint, but you must respect it when you
prepare data for fine-tuning.

## Sensor and bands (multispectral models)

The multispectral ("All") models were trained on **WorldView-3 (WV-3)**, 8
spectral bands, Gram–Schmidt pansharpened to ~0.31 m GSD, atmospherically
corrected (FLAASH), and converted from 11-bit to **8-bit (0–255 DN)**.

Band order (0-indexed) is fixed and part of the contract:

| Index | Band      | Wavelength (nm) |
|-------|-----------|-----------------|
| 0     | Coastal   | 400–450         |
| 1     | Blue      | 450–510         |
| 2     | Green     | 510–580         |
| 3     | Yellow    | 585–625         |
| 4     | Red       | 630–690         |
| 5     | Red Edge  | 705–745         |
| 6     | NIR-1     | 770–895         |
| 7     | NIR-2     | 860–1040        |

The spectral-index utility (`RSI_feature_creator`) and the stem band-map adapter
both assume this order (e.g. NDVI = (NIR1 − Red) = (band 6 − band 4)).

## The two contracts

### Multispectral (8-band), `modality='ms'`
- **Loader:** `LoadSingleRSImageFromFile` (GDAL), native band order preserved.
- **Values:** raw 8-bit DN, cast to float32. **No scaling, no contrast stretch,
  no /255.**
- **Data preprocessor:** `mean=[0]*8`, `std=[1]*8` (identity), `bgr_to_rgb=False`.
- **Tiles on disk:** multi-band `.tif`.

Consequence for data prep: if you tile WV-3 data for fine-tuning with the
`tile_pipeline`, set `APPLY_CONTRAST_STRETCH = False`. A stretched export
changes the input distribution the backbone learned and will degrade transfer.

### RGB, `modality='rgb'`
- **Loader:** `LoadImageFromFile` (OpenCV, BGR).
- **Values:** `bgr_to_rgb=True`, then ImageNet normalisation
  `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]`.
- **Tiles on disk:** 3-band `.tif`, `.png`, or `.jpg`.

For RGB, `APPLY_CONTRAST_STRETCH = True` in the tiler is acceptable because the
RGB path normalises with ImageNet statistics regardless.

## Why `bgr_to_rgb` is harmless at 8 bands
MMSegmentation's `SegDataPreProcessor` only reverses channel order when the
input has 3 channels. At 8 channels the flag is a silent no-op, which is why
some published configs left it `True`. The toolkit sets it `False` for the MS
path to remove the ambiguity entirely.

## Masks
Masks are **single-band integer-index rasters** (not colour-coded):
`0 = background`, `1 = palm`, `k = additional class`. `reduce_zero_label` is
always `False` — index 0 is a real, evaluated background class. The tiler writes
masks as `.png`; both loaders read masks the same way.

## What the loader does for you
`load_palm_model(checkpoint, config)` reads the checkpoint's stem (`in_channels`)
and head (`num_classes`) and **overrides the config to match**, then forces the
preprocessor above. This neutralises stale `num_classes=150` config text and
prevents using the wrong normalisation. You can still inspect a checkpoint with
`palmseg inspect <ckpt>`.


## RGB channel order in tiled inference (important)

The released RGB models were trained from OpenCV reads (BGR in memory) with the
data preprocessor's `bgr_to_rgb=True` restoring RGB before normalisation, so the
network learned on **RGB**. During large-raster inference the toolkit reads tiles
with rasterio in native band order (R, G, B) and therefore **reverses them to BGR
before handing them to the model**, so the preprocessor flips them back to RGB and
matches training exactly. This is handled automatically in
`palmseg.inference.tiled` for `modality='rgb'`; the multispectral path is never
reordered. If you write a custom inference loop, replicate this: feed RGB models a
BGR-ordered array (or set `bgr_to_rgb=False` and feed RGB directly).
