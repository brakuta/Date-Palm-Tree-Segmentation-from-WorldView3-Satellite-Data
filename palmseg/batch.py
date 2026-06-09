# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Batch processing over a folder of rasters.

Runs any of the three modes across every matching raster in an input directory,
with:
  * per-file fault isolation (one bad file does not abort the run),
  * resume (skip files whose outputs already exist), and
  * a summary report (and optional CSV) of counts and failures.

The model is loaded once and reused across all files. Extraction-only batch
mode needs no model.
"""
from __future__ import annotations

import csv
import glob
import logging
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger('palmseg.batch')

RASTER_EXTS = ('.tif', '.tiff', '.TIF', '.TIFF')


def _list_rasters(folder: str, pattern: Optional[str] = None) -> List[str]:
    if pattern:
        return sorted(glob.glob(osp.join(folder, pattern)))
    files: List[str] = []
    for ext in RASTER_EXTS:
        files.extend(glob.glob(osp.join(folder, f'*{ext}')))
    return sorted(set(files))


@dataclass
class BatchResult:
    processed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)   # (path, error)
    counts: dict = field(default_factory=dict)           # path -> tree count

    def summary(self) -> str:
        lines = [f'processed={len(self.processed)} '
                 f'skipped={len(self.skipped)} failed={len(self.failed)}']
        if self.counts:
            total = sum(self.counts.values())
            lines.append(f'total trees across batch: {total}')
        for p, e in self.failed:
            lines.append(f'  FAILED {osp.basename(p)}: {e}')
        return '\n'.join(lines)

    def write_csv(self, path: str) -> None:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'status', 'tree_count', 'error'])
            for p in self.processed:
                w.writerow([osp.basename(p), 'ok',
                            self.counts.get(p, ''), ''])
            for p in self.skipped:
                w.writerow([osp.basename(p), 'skipped', '', ''])
            for p, e in self.failed:
                w.writerow([osp.basename(p), 'failed', '', str(e)])


def batch_segment(model, in_dir, out_dir, *, pattern=None, resume=True,
                  tile_size=512, overlap=128, palm_class=1,
                  write_heatmap=True, summary_csv=None) -> BatchResult:
    from palmseg.pipeline import segment
    res = BatchResult()
    os.makedirs(out_dir, exist_ok=True)
    files = _list_rasters(in_dir, pattern)
    logger.info('batch_segment: %d raster(s) in %s', len(files), in_dir)
    for path in files:
        stem = osp.splitext(osp.basename(path))[0]
        label_out = osp.join(out_dir, f'{stem}_label.tif')
        if resume and osp.exists(label_out):
            res.skipped.append(path)
            continue
        try:
            segment(model, path, out_dir, tile_size=tile_size, overlap=overlap,
                    palm_class=palm_class, write_heatmap=write_heatmap,
                    stem=stem)
            res.processed.append(path)
        except Exception as exc:                       # isolate per file
            logger.exception('segment failed: %s', path)
            res.failed.append((path, repr(exc)))
    if summary_csv:
        res.write_csv(summary_csv)
    return res


def batch_extract(in_dir, out_dir, *, heatmap_dir=None, pattern=None,
                  resume=True, palm_class=1, method='watershed',
                  geometry='circle', summary_csv=None,
                  label_suffix='_label', heatmap_suffix='_prob',
                  **extract_kwargs) -> BatchResult:
    """Extract trees for every label raster in ``in_dir``.

    Heatmaps are matched by filename: for ``<stem><label_suffix>.tif`` the
    heatmap is looked up as ``<stem><heatmap_suffix>.tif`` in ``heatmap_dir``
    (or ``in_dir`` if not given). If no heatmap is found, extraction proceeds
    from the mask alone.
    """
    from palmseg.pipeline import extract
    res = BatchResult()
    os.makedirs(out_dir, exist_ok=True)
    heatmap_dir = heatmap_dir or in_dir
    files = _list_rasters(in_dir, pattern)
    logger.info('batch_extract: %d label raster(s) in %s', len(files), in_dir)
    for path in files:
        base = osp.splitext(osp.basename(path))[0]
        stem = base[:-len(label_suffix)] if base.endswith(label_suffix) else base
        out_vec = osp.join(out_dir, f'{stem}_trees.gpkg')
        if resume and osp.exists(out_vec):
            res.skipped.append(path)
            continue
        # locate matching heatmap if present
        hm = osp.join(heatmap_dir, f'{stem}{heatmap_suffix}.tif')
        hm = hm if osp.exists(hm) else None
        try:
            out = extract(path, out_vec, heatmap_path=hm, palm_class=palm_class,
                          method=method, geometry=geometry, **extract_kwargs)
            res.processed.append(path)
            res.counts[path] = out.count
        except Exception as exc:
            logger.exception('extract failed: %s', path)
            res.failed.append((path, repr(exc)))
    if summary_csv:
        res.write_csv(summary_csv)
    return res


def batch_segment_and_extract(model, in_dir, out_dir, *, pattern=None,
                              resume=True, tile_size=512, overlap=128,
                              palm_class=1, method='watershed',
                              geometry='circle', summary_csv=None,
                              **extract_kwargs) -> BatchResult:
    from palmseg.pipeline import segment_and_extract
    res = BatchResult()
    os.makedirs(out_dir, exist_ok=True)
    files = _list_rasters(in_dir, pattern)
    logger.info('batch_segment_and_extract: %d raster(s)', len(files))
    for path in files:
        stem = osp.splitext(osp.basename(path))[0]
        out_vec = osp.join(out_dir, f'{stem}_trees.gpkg')
        if resume and osp.exists(out_vec):
            res.skipped.append(path)
            continue
        try:
            _, ext = segment_and_extract(
                model, path, out_dir, out_vector_path=out_vec,
                tile_size=tile_size, overlap=overlap, palm_class=palm_class,
                method=method, geometry=geometry, **extract_kwargs)
            res.processed.append(path)
            res.counts[path] = ext.count
        except Exception as exc:
            logger.exception('segment-and-extract failed: %s', path)
            res.failed.append((path, repr(exc)))
    if summary_csv:
        res.write_csv(summary_csv)
    return res
