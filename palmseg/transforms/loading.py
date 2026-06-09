# Copyright (c) OpenMMLab. All rights reserved. Modifications (c) 2024 Apache-2.0.
"""Remote-sensing image loading for MMSegmentation 1.x.

Loads a multi-band raster as (H, W, C) in native band order, cast to float32
with no scaling. The multispectral models were trained on raw 8-bit DN with a
mean=0/std=1 preprocessor, so values are passed through unchanged.

Reading uses rasterio if available and falls back to osgeo.gdal. rasterio ships
its own GDAL and installs with pip, so a separate GDAL install is not required.
Both backends return identical arrays.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS

# Detect available backends at import time (cheap) but do not fail; the error
# (if any) is raised only when the transform is actually constructed/used.
try:
    import rasterio as _rasterio
except ImportError:
    _rasterio = None

try:
    from osgeo import gdal as _gdal
except ImportError:
    _gdal = None


def _read_native(path: str) -> np.ndarray:
    """Return (H, W, C) array in native band order using the best backend."""
    if _rasterio is not None:
        with _rasterio.open(path) as ds:
            arr = ds.read()                       # (C, H, W)
    elif _gdal is not None:
        ds = _gdal.Open(path)
        if ds is None:
            raise FileNotFoundError(f'GDAL could not open: {path}')
        arr = ds.ReadAsArray()                    # (C, H, W) or (H, W)
    else:
        raise RuntimeError(
            'Neither rasterio nor osgeo.gdal is available. Install rasterio '
            '(recommended: `pip install rasterio` or `conda install -c '
            'conda-forge rasterio`); it bundles GDAL and needs no separate '
            'GDAL install.')
    if arr.ndim == 2:                             # single-band -> (1, H, W)
        arr = arr[np.newaxis, ...]
    return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))  # (H, W, C)


@TRANSFORMS.register_module()
class LoadSingleRSImageFromFile(BaseTransform):
    """Load a (multi-band) remote-sensing image from file.

    Required Keys:
        - img_path
    Modified Keys:
        - img        (np.ndarray, H x W x C, float32 by default)
        - img_shape
        - ori_shape

    Args:
        to_float32 (bool): Cast to float32. Default True (matches training).
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32
        if _rasterio is None and _gdal is None:
            raise RuntimeError(
                'LoadSingleRSImageFromFile requires a raster backend but '
                'neither rasterio nor osgeo.gdal is installed. Install '
                'rasterio: `pip install rasterio` (recommended; bundles GDAL) '
                'or `conda install -c conda-forge rasterio`.')

    def transform(self, results: Dict) -> Dict:
        img = _read_native(results['img_path'])
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        backend = 'rasterio' if _rasterio is not None else (
            'gdal' if _gdal is not None else 'none')
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f'backend={backend})')
