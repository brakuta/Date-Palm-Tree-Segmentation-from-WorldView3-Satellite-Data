#!/usr/bin/env python3
# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Generate a small synthetic georeferenced scene to try the toolkit end to end.

This writes:
  demo/scene_rgb.tif        a 3-band RGB scene with palm-like blobs
  demo/scene_label.tif      a class-label raster (0 background, 1 palm)
  demo/scene_prob.tif       a palm-probability heatmap in [0,1]

You can then run individual-tree extraction (mode b) directly on the label +
heatmap WITHOUT a GPU or model:

    palmseg extract --label demo/scene_label.tif --heatmap demo/scene_prob.tif \
                    --out demo/scene_trees.gpkg

This is purely for learning the workflow; it is not a substitute for real data.
"""
from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main(out_dir: str = 'demo', size: int = 512, n_trees: int = 40,
         seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    H = W = size
    px = 0.31  # WV-3-like GSD in metres
    transform = from_origin(400000.0, 2700000.0, px, px)
    crs = 'EPSG:32640'

    prob = np.zeros((H, W), np.float32)
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n_trees):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = rng.integers(8, 16)            # crown radius in px (~2.5-5 m)
        blob = np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1)
        prob = np.maximum(prob, blob.astype(np.float32))

    label = (prob > 0.3).astype(np.uint8)

    # RGB scene: greenish where palm, sandy elsewhere, plus mild noise.
    rgb = np.zeros((3, H, W), np.uint8)
    sand = np.array([194, 178, 128], np.float32)
    green = np.array([34, 110, 34], np.float32)
    for b in range(3):
        base = np.where(label == 1, green[b], sand[b])
        noise = rng.normal(0, 8, (H, W))
        rgb[b] = np.clip(base + noise, 0, 255).astype(np.uint8)

    def _write(path, arr, dtype, count):
        prof = dict(driver='GTiff', height=H, width=W, count=count,
                    dtype=dtype, crs=crs, transform=transform,
                    compress='deflate')
        with rasterio.open(path, 'w', **prof) as d:
            if count == 1:
                d.write(arr.astype(dtype), 1)
            else:
                d.write(arr.astype(dtype))

    _write(os.path.join(out_dir, 'scene_rgb.tif'), rgb, 'uint8', 3)
    _write(os.path.join(out_dir, 'scene_label.tif'), label, 'uint8', 1)
    _write(os.path.join(out_dir, 'scene_prob.tif'), prob, 'float32', 1)
    print(f'Wrote demo data to {out_dir}/ '
          f'(scene_rgb.tif, scene_label.tif, scene_prob.tif); '
          f'{int(label.sum())} palm pixels, ~{n_trees} trees.')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='demo')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--n-trees', type=int, default=40)
    main(**vars(ap.parse_args()))
