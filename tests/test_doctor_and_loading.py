# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for the doctor diagnostics and the rasterio-first RS loader."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def test_doctor_runs_and_returns_int():
    from palmseg import doctor
    rc = doctor.run()
    assert rc in (0, 1)


def test_doctor_detects_raster_backend():
    from palmseg.doctor import _check_raster_backend
    c = _check_raster_backend()
    # rasterio is a hard dependency of the test env, so this must pass
    assert c.ok
    assert 'rasterio' in c.detail or 'gdal' in c.detail


def test_doctor_cuda_section_no_crash():
    from palmseg.doctor import _cuda_pytorch_check
    checks = _cuda_pytorch_check()
    assert len(checks) >= 1
    assert all(hasattr(c, 'ok') for c in checks)


def test_rs_loader_reads_native_band_order(tmp_path):
    """The multispectral loader must preserve band order and not rescale."""
    # Build an 8-band raster with distinct per-band constants.
    H = W = 32
    arr = np.zeros((8, H, W), np.float32)
    for b in range(8):
        arr[b] = (b + 1) * 10            # band b -> value 10*(b+1)
    path = str(tmp_path / 'ms.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=8,
                       dtype='float32', crs='EPSG:32640',
                       transform=from_origin(0, 100, 0.5, 0.5)) as d:
        d.write(arr)

    from palmseg.transforms.loading import _read_native
    img = _read_native(path)              # (H, W, C)
    assert img.shape == (H, W, 8)
    # band order preserved, values unscaled
    for b in range(8):
        assert np.allclose(img[:, :, b], (b + 1) * 10)


def test_rs_loader_single_band(tmp_path):
    H = W = 16
    path = str(tmp_path / 's.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs='EPSG:32640',
                       transform=from_origin(0, 100, 0.5, 0.5)) as d:
        d.write((np.ones((H, W)) * 7).astype('uint8'), 1)
    from palmseg.transforms.loading import _read_native
    img = _read_native(path)
    assert img.shape == (H, W, 1)
    assert np.allclose(img[:, :, 0], 7)
