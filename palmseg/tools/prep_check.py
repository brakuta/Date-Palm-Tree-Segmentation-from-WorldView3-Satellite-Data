# Copyright (c) 2024. Licensed under Apache-2.0.
"""Validate a prepared dataset before fine-tuning.

Catches the most common fine-tune failures:
  * image/mask filename mismatch between img_dir and ann_dir,
  * masks that are not single-band integer-index rasters,
  * mask class values outside [0, num_classes-1] (or stray colour-coding),
  * band-count mismatch between the imagery and the chosen model modality,
  * the modality/stretch contract violation (MS imagery must be raw, not a
    stretched 3-band export).

Run via the CLI: ``palmseg prep-check --data-root data/palm_ms --modality ms``.
"""
from __future__ import annotations

import os
import os.path as osp
from collections import Counter
from typing import List, Tuple

import numpy as np


def _list_split(data_root: str, split: str) -> Tuple[str, str, List[str], List[str]]:
    img_dir = osp.join(data_root, 'img_dir', split)
    ann_dir = osp.join(data_root, 'ann_dir', split)
    imgs = sorted(os.listdir(img_dir)) if osp.isdir(img_dir) else []
    anns = sorted(os.listdir(ann_dir)) if osp.isdir(ann_dir) else []
    return img_dir, ann_dir, imgs, anns


def _stem(name: str) -> str:
    return osp.splitext(name)[0]


def check_dataset(data_root: str,
                  modality: str = 'ms',
                  num_classes: int = 2,
                  expected_bands: int = None,
                  max_mask_sample: int = 50) -> List[str]:
    """Return a list of problems found (empty list == OK)."""
    problems: List[str] = []
    if modality == 'ms' and expected_bands is None:
        expected_bands = 8
    if modality == 'rgb' and expected_bands is None:
        expected_bands = 3

    try:
        import rasterio
    except ImportError:
        problems.append('rasterio is required for prep-check.')
        return problems

    found_any = False
    for split in ('train', 'val', 'test'):
        img_dir, ann_dir, imgs, anns = _list_split(data_root, split)
        if not imgs and not anns:
            continue
        found_any = True

        img_stems = {_stem(x) for x in imgs}
        ann_stems = {_stem(x) for x in anns}
        missing_masks = img_stems - ann_stems
        missing_imgs = ann_stems - img_stems
        if missing_masks:
            problems.append(
                f'[{split}] {len(missing_masks)} image(s) without a matching '
                f'mask (e.g. {sorted(missing_masks)[:3]}).')
        if missing_imgs:
            problems.append(
                f'[{split}] {len(missing_imgs)} mask(s) without a matching '
                f'image (e.g. {sorted(missing_imgs)[:3]}).')

        # Inspect a sample of images for band count / dtype.
        for name in imgs[:max_mask_sample]:
            p = osp.join(img_dir, name)
            try:
                with rasterio.open(p) as src:
                    if src.count != expected_bands:
                        problems.append(
                            f'[{split}] {name}: {src.count} band(s) but '
                            f'modality={modality} expects {expected_bands}.')
                        break
                    if modality == 'ms' and src.dtypes[0] != 'uint8':
                        # MS contract is 8-bit 0-255 DN. Float/16-bit suggests
                        # a different normalisation than the released models.
                        problems.append(
                            f'[{split}] {name}: dtype {src.dtypes[0]}; the '
                            f'released MS models expect 8-bit (0-255) DN. If '
                            f'you re-stretched, retrain or adjust the '
                            f'preprocessor mean/std accordingly.')
                        break
            except Exception as exc:
                problems.append(f'[{split}] cannot open image {name}: {exc}')
                break

        # Inspect a sample of masks for single-band + class range.
        seen_values: Counter = Counter()
        for name in anns[:max_mask_sample]:
            p = osp.join(ann_dir, name)
            try:
                with rasterio.open(p) as src:
                    if src.count != 1:
                        problems.append(
                            f'[{split}] mask {name} has {src.count} bands; '
                            f'masks must be single-band index rasters.')
                        break
                    band = src.read(1)
            except Exception:
                # Fall back to PIL for .png masks.
                try:
                    from PIL import Image
                    band = np.array(Image.open(p))
                    if band.ndim != 2:
                        problems.append(
                            f'[{split}] mask {name} is not single-channel '
                            f'(shape {band.shape}); expected an index raster.')
                        break
                except Exception as exc:
                    problems.append(f'[{split}] cannot open mask {name}: {exc}')
                    break
            vals = np.unique(band)
            seen_values.update(vals.tolist())
            bad = vals[(vals != 255) & (vals >= num_classes)]
            if bad.size > 0:
                problems.append(
                    f'[{split}] mask {name}: values {bad.tolist()} exceed '
                    f'num_classes-1={num_classes-1} (255 is allowed as ignore). '
                    f'Masks must be class indices, not colour-coded.')
                break
        if seen_values:
            top = ', '.join(f'{v}:{c}' for v, c in seen_values.most_common(6))
            print(f'[{split}] mask value histogram (sample): {top}')

    if not found_any:
        problems.append(
            f'No img_dir/<split> + ann_dir/<split> folders found under '
            f'{data_root}. Expected layout: '
            f'{data_root}/img_dir/train, {data_root}/ann_dir/train, ...')
    return problems
