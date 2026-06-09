# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for batch extraction (no GPU/model needed)."""
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from palmseg.batch import batch_extract

CRS = 'EPSG:32640'
TR = from_origin(0, 200, 0.31, 0.31)


def _disk(cy, cx, r, H, W):
    yy, xx = np.ogrid[:H, :W]
    return np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1)


def _make_scene(d, i, H=120, W=120):
    prob = np.maximum(_disk(40, 40, 15, H, W),
                      _disk(80, 80, 18, H, W)).astype('float32')
    lab = (prob > 0.2).astype('uint8')
    with rasterio.open(f'{d}/scene{i}_label.tif', 'w', driver='GTiff',
                       height=H, width=W, count=1, dtype='uint8', crs=CRS,
                       transform=TR) as f:
        f.write(lab, 1)
    with rasterio.open(f'{d}/scene{i}_prob.tif', 'w', driver='GTiff',
                       height=H, width=W, count=1, dtype='float32', crs=CRS,
                       transform=TR) as f:
        f.write(prob, 1)


def test_batch_extract_with_fault_isolation_and_resume(tmp_path):
    d = tmp_path / 'in'
    o = tmp_path / 'out'
    d.mkdir()
    for i in range(3):
        _make_scene(str(d), i)
    # a corrupt file must not abort the batch
    (d / 'broken_label.tif').write_text('not a raster')

    res = batch_extract(str(d), str(o), pattern='*_label.tif',
                        heatmap_dir=str(d), min_area_px=20,
                        min_seed_distance=6,
                        summary_csv=str(o / 'summary.csv'))
    assert len(res.processed) == 3
    assert len(res.failed) == 1
    assert sum(res.counts.values()) == 6
    assert os.path.exists(str(o / 'summary.csv'))

    # resume: rerun skips the 3 completed
    res2 = batch_extract(str(d), str(o), pattern='*_label.tif',
                         heatmap_dir=str(d), min_area_px=20,
                         min_seed_distance=6)
    assert len(res2.skipped) == 3
    assert len(res2.processed) == 0


def test_batch_extract_mask_only(tmp_path):
    """No heatmaps present -> extraction still runs from masks alone."""
    d = tmp_path / 'in'
    o = tmp_path / 'out'
    d.mkdir()
    for i in range(2):
        H = W = 120
        prob = _disk(60, 60, 20, H, W).astype('float32')
        lab = (prob > 0.2).astype('uint8')
        with rasterio.open(f'{d}/s{i}_label.tif', 'w', driver='GTiff',
                           height=H, width=W, count=1, dtype='uint8',
                           crs=CRS, transform=TR) as f:
            f.write(lab, 1)
    res = batch_extract(str(d), str(o), pattern='*_label.tif',
                        min_area_px=20)
    assert len(res.processed) == 2
    assert all(c >= 1 for c in res.counts.values())
