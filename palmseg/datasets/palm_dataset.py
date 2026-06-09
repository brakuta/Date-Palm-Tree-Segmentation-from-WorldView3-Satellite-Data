# Copyright (c) 2024. Licensed under Apache-2.0.
"""Date palm semantic-segmentation dataset for MMSegmentation 1.x.

Two-class by default (0 background, 1 palm); override ``metainfo`` for N-class
use. Masks are single-band index rasters with ``reduce_zero_label=False`` (0 is
an evaluated background class). Image and mask suffixes default to ``.tif``.
"""
from __future__ import annotations

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


# Default 2-class setup as published. Override ``metainfo`` in a fine-tune
# config to add classes, e.g.
#   metainfo=dict(classes=('background', 'palm', 'ghaf'),
#                 palette=[[0,0,0],[255,0,37],[0,128,0]])
PALM_CLASSES = ('background', 'palm')
PALM_PALETTE = [[0, 0, 0], [255, 0, 37]]


@DATASETS.register_module()
class PalmDataset(BaseSegDataset):
    """Palm crown semantic-segmentation dataset.

    Args:
        img_suffix (str): Image file suffix. ``.tif`` for multispectral and
            for the RGB GeoTIFF tiles used in this project. Override to
            ``.jpg``/``.png`` only if your RGB tiles are stored as such.
        seg_map_suffix (str): Mask file suffix. ``.tif`` (single-band index
            raster) in this project.
        reduce_zero_label (bool): Must stay False; 0 is a real background
            class that is evaluated.
    """

    METAINFO = dict(classes=PALM_CLASSES, palette=PALM_PALETTE)

    def __init__(self,
                 img_suffix: str = '.tif',
                 seg_map_suffix: str = '.tif',
                 reduce_zero_label: bool = False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


