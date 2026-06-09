# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for tiled inference stitching (mock model; no GPU/MMSeg install)."""
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

torch = pytest.importorskip('torch')
from palmseg.inference import tiled


class _FD:
    def __init__(self, a):
        self.data = a


class _FR:
    def __init__(self, logits, label):
        self.seg_logits = _FD(torch.tensor(logits))
        self.pred_sem_seg = _FD(torch.tensor(label)[None])


def _install_fake_mmseg(infer_fn):
    mm = types.ModuleType('mmseg')
    ap = types.ModuleType('mmseg.apis')
    ap.inference_model = infer_fn
    sys.modules['mmseg'] = mm
    sys.modules['mmseg.apis'] = ap


def _write(path, arr, nodata=None, crs='EPSG:32640'):
    c, h, w = arr.shape
    prof = dict(driver='GTiff', height=h, width=w, count=c, dtype=arr.dtype,
                crs=crs, transform=from_origin(0, 1000, 0.5, 0.5))
    if nodata is not None:
        prof['nodata'] = nodata
    with rasterio.open(path, 'w', **prof) as d:
        d.write(arr)


class _M:
    cfg_info = {'modality': 'rgb', 'in_channels': 3, 'num_classes': 2}


def test_label_heatmap_consistency(tmp_path):
    calls = {'n': 0}

    def infer(model, img):
        h, w = img.shape[:2]
        calls['n'] += 1
        palm = np.full((h, w), 2.0 if calls['n'] % 2 else -2.0, np.float32)
        return _FR(np.stack([np.zeros((h, w), np.float32), palm], 0),
                   (palm > 0).astype(np.int64))
    _install_fake_mmseg(infer)
    src = str(tmp_path / 's.tif')
    _write(src, (np.ones((3, 200, 200)) * 100).astype('uint8'))
    lp, pp = str(tmp_path / 'l.tif'), str(tmp_path / 'p.tif')
    tiled.infer_raster(_M(), src, lp, pp,
                       tiled.TileInferConfig(tile_size=128, overlap=64))
    lab = rasterio.open(lp).read(1)
    prob = rasterio.open(pp).read(1)
    # label==palm must agree with prob>0.5 everywhere (consistency by design)
    assert int(((lab == 1) != (prob > 0.5)).sum()) == 0


def test_nodata_excluded(tmp_path):
    def infer(model, img):
        h, w = img.shape[:2]
        return _FR(np.stack([np.zeros((h, w), np.float32),
                             np.full((h, w), 2.0, np.float32)], 0),
                   np.ones((h, w), np.int64))
    _install_fake_mmseg(infer)
    arr = np.ones((3, 200, 200), 'uint8') * 100
    arr[:, :40, :40] = 0
    src = str(tmp_path / 'nd.tif')
    _write(src, arr, nodata=0)
    lp, pp = str(tmp_path / 'l.tif'), str(tmp_path / 'p.tif')
    tiled.infer_raster(_M(), src, lp, pp,
                       tiled.TileInferConfig(tile_size=128, overlap=64))
    lab = rasterio.open(lp).read(1)
    assert int(lab[:40, :40].sum()) == 0   # nodata corner stays background


def test_rgb_channel_reversal(tmp_path):
    seen = []

    def infer(model, img):
        h, w = img.shape[:2]
        seen.append((float(img[:, :, 0].mean()), float(img[:, :, 2].mean())))
        return _FR(np.stack([np.zeros((h, w), np.float32),
                             np.full((h, w), 2.0, np.float32)], 0),
                   np.ones((h, w), np.int64))
    _install_fake_mmseg(infer)
    arr = np.zeros((3, 140, 140), 'uint8')
    arr[0] = 10; arr[1] = 20; arr[2] = 30   # R,G,B
    src = str(tmp_path / 'o.tif')
    _write(src, arr)
    tiled.infer_raster(_M(), src, str(tmp_path / 'l.tif'),
                       str(tmp_path / 'p.tif'),
                       tiled.TileInferConfig(tile_size=140, overlap=0))
    # RGB reversed to BGR for the model: ch0 sees B(30), ch2 sees R(10)
    assert abs(seen[0][0] - 30) < 1 and abs(seen[0][1] - 10) < 1


def test_ms_no_reversal(tmp_path):
    seen = []

    def infer(model, img):
        h, w = img.shape[:2]
        seen.append((float(img[:, :, 0].mean()), float(img[:, :, 7].mean())))
        return _FR(np.stack([np.zeros((h, w), np.float32),
                             np.full((h, w), 1.0, np.float32)], 0),
                   np.ones((h, w), np.int64))
    _install_fake_mmseg(infer)
    arr = np.zeros((8, 140, 140), 'uint8')
    arr[0] = 10; arr[7] = 80
    src = str(tmp_path / 'm.tif')
    _write(src, arr)

    class M8:
        cfg_info = {'modality': 'ms', 'in_channels': 8, 'num_classes': 2}
    tiled.infer_raster(M8(), src, str(tmp_path / 'l.tif'),
                       str(tmp_path / 'p.tif'),
                       tiled.TileInferConfig(tile_size=140, overlap=0))
    assert abs(seen[0][0] - 10) < 1 and abs(seen[0][1] - 80) < 1


def test_memory_guard():
    with pytest.raises(MemoryError):
        tiled._check_memory(200000, 200000, 2, 8.0)


def test_heatmap_optional(tmp_path):
    def infer(model, img):
        h, w = img.shape[:2]
        return _FR(np.stack([np.zeros((h, w), np.float32),
                             np.full((h, w), 2.0, np.float32)], 0),
                   np.ones((h, w), np.int64))
    _install_fake_mmseg(infer)
    src = str(tmp_path / 's.tif')
    _write(src, (np.ones((3, 140, 140)) * 100).astype('uint8'))
    stats = tiled.infer_raster(
        _M(), src, str(tmp_path / 'l.tif'), None,
        tiled.TileInferConfig(tile_size=140, overlap=0, write_heatmap=False))
    assert stats['prob_path'] is None
