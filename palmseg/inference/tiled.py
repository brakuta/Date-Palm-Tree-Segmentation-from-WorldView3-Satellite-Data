# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tiled semantic inference over large georeferenced rasters.

Slides a window over the input raster, runs the model per tile, and stitches the
results into a class-label raster and, optionally, a palm-probability heatmap.

Properties:
  - Windowed reads; the source raster is not loaded in full.
  - Overlapping tiles are blended with an edge-tapered weight to avoid seams.
  - The output label is the argmax of the blended per-class probabilities, so it
    is consistent with the exported heatmap.
  - Pixels equal to the raster nodata value are excluded and left as background.
  - Multispectral input is passed through in native band order (the model's
    preprocessor applies mean=0/std=1). RGB input is reversed to BGR so the
    preprocessor's bgr_to_rgb restores the training order.
  - Full-resolution accumulators scale with image size; a configurable budget
    raises a clear error rather than an opaque MemoryError on oversized inputs.

Model modality and class count are read from ``model.cfg_info`` (set by
``palmseg.loader.load_palm_model``).
"""
from __future__ import annotations

import logging
import os
import os.path as osp
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rasterio
from rasterio.windows import Window

logger = logging.getLogger('palmseg.inference')

# mmseg is imported lazily inside infer_raster so this module (and the
# extraction-only mode) can be imported without MMSegmentation installed.


@dataclass
class TileInferConfig:
    tile_size: int = 512
    overlap: int = 128
    palm_class: int = 1
    write_heatmap: bool = True
    device_empty_cache_on_oom: bool = True
    modality: Optional[str] = None          # 'ms' | 'rgb'; else from cfg_info
    edge_taper: float = 0.1                 # min weight at the very tile edge
    max_accumulator_gb: float = 8.0         # memory budget guard
    skip_nodata_tiles: bool = True


def _num_classes_from_logits(seg_logits) -> int:
    return int(seg_logits.shape[0])


def _read_tile(ds, x: int, y: int, ts: int):
    """Windowed read -> (HWC float32 tile, valid-pixel mask HxW bool)."""
    window = Window(x, y, ts, ts)
    arr = ds.read(window=window, boundless=True, fill_value=0)  # (C, ts, ts)
    tile = np.ascontiguousarray(np.transpose(arr, (1, 2, 0))).astype(np.float32)
    # Valid mask from nodata, if defined; else everything in-bounds is valid.
    nodata = ds.nodata
    if nodata is not None:
        valid = ~np.all(arr == nodata, axis=0)
    else:
        valid = np.ones((arr.shape[1], arr.shape[2]), dtype=bool)
    return tile, valid


def _edge_weight(ts: int, overlap: int, taper: float) -> np.ndarray:
    w = np.ones(ts, np.float32)
    if overlap > 0:
        ramp = np.linspace(taper, 1.0, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w)


def _check_memory(H: int, W: int, n_classes: int, budget_gb: float):
    # prob accumulator (n_classes) + weight + label : float32/uint8.
    bytes_needed = (H * W) * (4 * (n_classes + 1) + 1)
    gb = bytes_needed / 1e9
    if gb > budget_gb:
        raise MemoryError(
            f'Stitching {W}x{H} at {n_classes} classes needs ~{gb:.1f} GB '
            f'(budget {budget_gb} GB). Process the raster in sub-regions or '
            f'pre-tile it (e.g. gdal_retile.py) and run inference per tile, '
            f'or raise TileInferConfig.max_accumulator_gb if you have the RAM.')


def infer_raster(model,
                 raster_path: str,
                 out_label_path: str,
                 out_prob_path: Optional[str] = None,
                 cfg: Optional[TileInferConfig] = None) -> dict:
    """Run tiled inference over one raster and write label (+ heatmap) GeoTIFFs.

    Returns a stats dict. ``out_prob_path`` may be None to skip the heatmap.
    """
    cfg = cfg or TileInferConfig()
    if cfg.tile_size <= 0:
        raise ValueError(f'tile_size must be positive, got {cfg.tile_size}')
    if cfg.overlap < 0:
        raise ValueError(f'overlap must be >= 0, got {cfg.overlap}')
    modality = cfg.modality or getattr(model, 'cfg_info', {}).get('modality',
                                                                  'ms')
    ts, stride = cfg.tile_size, cfg.tile_size - cfg.overlap
    if stride <= 0:
        raise ValueError('overlap must be smaller than tile_size')
    if not osp.isfile(raster_path):
        raise FileNotFoundError(f'Input raster not found: {raster_path}')
    # Heavy import deferred until inputs are known-good, so bad paths and
    # bad configs fail instantly even before MMSegmentation loads.
    from mmseg.apis import inference_model

    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        transform, crs = src.transform, src.crs

    # Probe num_classes from one inference so accumulators are sized correctly.
    n_classes = None
    tile_weight = _edge_weight(ts, cfg.overlap, cfg.edge_taper)

    prob_acc = None
    weight_acc = np.zeros((H, W), dtype=np.float32)
    n_tiles = 0

    ds = rasterio.open(raster_path)
    try:
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile, valid = _read_tile(ds, x, y, ts)
                if cfg.skip_nodata_tiles and not valid.any():
                    continue

                # RGB models were trained from OpenCV BGR reads with
                # bgr_to_rgb=True. rasterio returns native (R,G,B) order, so
                # reverse to BGR here; the preprocessor flips it back to RGB.
                # Multispectral input is left in native order.
                model_input = tile
                if modality == 'rgb' and tile.shape[2] == 3:
                    model_input = tile[:, :, ::-1].copy()

                try:
                    result = inference_model(model, model_input)
                except RuntimeError as exc:
                    if ('out of memory' in str(exc).lower()
                            and cfg.device_empty_cache_on_oom):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Retry with the SAME prepared input. Re-running the
                        # raw `tile` here would feed RGB models un-reversed
                        # channels and silently corrupt retried tiles.
                        result = inference_model(model, model_input)
                    else:
                        raise

                logits = result.seg_logits.data            # (C, ts, ts)
                if n_classes is None:
                    n_classes = _num_classes_from_logits(logits)
                    if cfg.write_heatmap and cfg.palm_class >= n_classes:
                        raise ValueError(
                            f'palm_class={cfg.palm_class} is out of range for '
                            f'a {n_classes}-class model (valid: 0..'
                            f'{n_classes - 1}).')
                    _check_memory(H, W, n_classes, cfg.max_accumulator_gb)
                    prob_acc = np.zeros((n_classes, H, W), dtype=np.float32)

                import torch
                probs = torch.softmax(logits.float(), dim=0)\
                    .detach().cpu().numpy()                # (C, ts, ts)

                h, w = min(ts, H - y), min(ts, W - x)
                tw = tile_weight[:h, :w] * valid[:h, :w]   # zero out nodata px
                prob_acc[:, y:y+h, x:x+w] += probs[:, :h, :w] * tw[None]
                weight_acc[y:y+h, x:x+w] += tw
                n_tiles += 1
    finally:
        ds.close()

    if n_classes is None:
        raise RuntimeError('No valid tiles were processed (all nodata?).')

    nz = weight_acc > 0
    # Normalise blended per-class probability IN PLACE where covered. This
    # avoids allocating a second (C, H, W) float32 array, which would double
    # the peak memory relative to the _check_memory budget.
    prob_acc[:, nz] /= weight_acc[nz]
    # Label is the argmax of the blended probabilities, consistent with the
    # exported heatmap. Uncovered pixels stay background.
    label = np.zeros((H, W), dtype=np.uint8)
    label[nz] = np.argmax(prob_acc[:, nz], axis=0).astype(np.uint8)

    _write_single_band(out_label_path, label, profile, transform, crs,
                       'uint8')
    out_prob_written = None
    if cfg.write_heatmap and out_prob_path is not None:
        palm_prob = prob_acc[cfg.palm_class]
        _write_single_band(out_prob_path, palm_prob, profile, transform, crs,
                           'float32')
        out_prob_written = out_prob_path

    logger.info('infer_raster: %s -> %d tiles, %d classes, modality=%s',
                raster_path, n_tiles, n_classes, modality)
    return dict(tiles=n_tiles, height=H, width=W, num_classes=n_classes,
                modality=modality, palm_class=cfg.palm_class,
                label_path=out_label_path, prob_path=out_prob_written)


def _write_single_band(path, arr, src_profile, transform, crs, dtype):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    profile = src_profile.copy()
    profile.update(count=1, dtype=dtype, transform=transform, crs=crs,
                   compress='deflate', tiled=True, blockxsize=256,
                   blockysize=256)
    profile.pop('nodata', None)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype(dtype), 1)
