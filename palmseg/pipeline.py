# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""High-level orchestration for the three independent operating modes.

The toolkit exposes three capabilities that can each run on their own:

  (a) segment  : run a model over a raster -> class-label raster (+ optional
                 palm-probability heatmap raster). No instance extraction.
  (b) extract  : individual-tree extraction from EXISTING rasters you already
                 have -> crown polygons + counts. Accepts a label/mask raster
                 and an OPTIONAL probability heatmap; with no heatmap it seeds
                 from a distance transform, so it runs on a binary mask alone.
  (c) segment_and_extract : (a) then (b) in one call.

These are thin, well-logged orchestrators over the building blocks in
``palmseg.inference.tiled`` (segmentation) and ``palmseg.postprocess``
(extraction + vectorisation). Each is independently importable and testable;
extraction has no dependency on the model or on MMSegmentation.
"""
from __future__ import annotations

import logging
import os
import os.path as osp
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger('palmseg.pipeline')


@dataclass
class SegmentOutputs:
    label_path: str
    prob_path: Optional[str]
    tiles: int
    modality: str


@dataclass
class ExtractOutputs:
    vector_path: str
    count: int
    label_used: str
    heatmap_used: Optional[str]
    method: str


# ---------------------------------------------------------------------------
# (a) segmentation only
# ---------------------------------------------------------------------------
def segment(model,
            raster_path: str,
            out_dir: str,
            tile_size: int = 512,
            overlap: int = 128,
            palm_class: int = 1,
            write_heatmap: bool = True,
            stem: Optional[str] = None) -> SegmentOutputs:
    """Run tiled semantic inference. Writes a label raster and, optionally, the
    palm-class probability heatmap (needed later only if you want the most
    precise tree extraction).

    Returns paths and stats. Does not extract trees.
    """
    from palmseg.inference.tiled import infer_raster, TileInferConfig
    if not osp.isfile(raster_path):
        raise FileNotFoundError(f'Input raster not found: {raster_path}')
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or osp.splitext(osp.basename(raster_path))[0]
    label_path = osp.join(out_dir, f'{stem}_label.tif')
    prob_path = osp.join(out_dir, f'{stem}_prob.tif') if write_heatmap else None

    cfg = TileInferConfig(tile_size=tile_size, overlap=overlap,
                          palm_class=palm_class, write_heatmap=write_heatmap)
    stats = infer_raster(model, raster_path, label_path, prob_path, cfg)
    logger.info('segment: %s -> label=%s heatmap=%s (%d tiles)',
                raster_path, label_path, prob_path, stats['tiles'])
    return SegmentOutputs(label_path=label_path, prob_path=prob_path,
                          tiles=stats['tiles'], modality=stats['modality'])


# ---------------------------------------------------------------------------
# (b) extraction only (from existing rasters)
# ---------------------------------------------------------------------------
def extract(label_or_mask_path: str,
            out_vector_path: str,
            heatmap_path: Optional[str] = None,
            palm_class: int = 1,
            method: str = 'watershed',
            geometry: str = 'circle',
            blur_ksize: int = 5,
            prob_threshold: float = 0.5,
            min_area_px: int = 9,
            min_circularity: float = 0.0,
            min_seed_distance: int = 0,
            dedup_iou: float = 0.0,
            write_instance_raster: Optional[str] = None) -> ExtractOutputs:
    """Extract individual trees from existing rasters.

    Args:
        label_or_mask_path: a class-label raster (palm == ``palm_class``) or a
            binary mask raster (palm == 1).
        out_vector_path: output .gpkg or .geojson.
        heatmap_path: OPTIONAL palm-probability raster. If omitted, seeds are
            derived from a distance transform of the mask (works on a binary
            mask alone, slightly less precise on touching crowns).
        palm_class: class value treated as palm in the label raster.
        method: 'watershed' (default) or 'voronoi'.
        geometry: 'circle' (area-preserving) or 'polygon' (mask outline).
        write_instance_raster: optional path to also write the int32 instance
            label raster (georeferenced like the input).

    Returns:
        ExtractOutputs with the vector path and tree count.
    """
    import rasterio
    if not osp.isfile(label_or_mask_path):
        raise FileNotFoundError(f'Label/mask raster not found: '
                                f'{label_or_mask_path}')
    if heatmap_path and not osp.isfile(heatmap_path):
        raise FileNotFoundError(f'Heatmap raster not found: {heatmap_path}')
    from palmseg.postprocess.individual_trees import extract_trees
    from palmseg.postprocess.vectorize import (instances_to_gdf, save_vector,
                                                deduplicate_overlaps)

    with rasterio.open(label_or_mask_path) as src:
        label = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    prob = None
    if heatmap_path:
        with rasterio.open(heatmap_path) as src:
            prob = src.read(1).astype('float32')
        if prob.shape != label.shape:
            raise ValueError(
                f'heatmap shape {prob.shape} != label shape {label.shape}')

    # A label raster uses palm_class; a binary mask uses 1. Treat both: any
    # pixel == palm_class OR (max==1 and ==1).
    palm_mask = (label == palm_class)
    if not palm_mask.any() and label.max() == 1:
        palm_mask = (label == 1)
    if not palm_mask.any():
        logger.warning(
            'extract: no pixels of class %d (or 1) in %s; values present: %s. '
            'An empty layer will be written - check --palm-class if this is '
            'unexpected.', palm_class, label_or_mask_path,
            sorted(int(v) for v in set(label.ravel()[::max(1, label.size // 100000)])))

    res = extract_trees(palm_mask, prob, method=method, blur_ksize=blur_ksize,
                        prob_threshold=prob_threshold, min_area_px=min_area_px,
                        min_circularity=min_circularity,
                        min_seed_distance=min_seed_distance)

    gdf = instances_to_gdf(res.trees, transform, crs, geometry=geometry)
    if dedup_iou and dedup_iou > 0:
        gdf = deduplicate_overlaps(gdf, iou_threshold=dedup_iou)
    save_vector(gdf, out_vector_path)

    if write_instance_raster:
        prof = profile.copy()
        prof.update(count=1, dtype='int32', compress='deflate')
        prof.pop('nodata', None)
        os.makedirs(osp.dirname(osp.abspath(write_instance_raster)),
                    exist_ok=True)
        with rasterio.open(write_instance_raster, 'w', **prof) as dst:
            dst.write(res.label_img.astype('int32'), 1)

    logger.info('extract: %s (heatmap=%s) -> %d trees -> %s',
                label_or_mask_path, bool(heatmap_path), res.count,
                out_vector_path)
    return ExtractOutputs(vector_path=out_vector_path, count=res.count,
                          label_used=label_or_mask_path,
                          heatmap_used=heatmap_path, method=method)


# ---------------------------------------------------------------------------
# (c) end-to-end
# ---------------------------------------------------------------------------
def segment_and_extract(model,
                        raster_path: str,
                        out_dir: str,
                        out_vector_path: Optional[str] = None,
                        tile_size: int = 512,
                        overlap: int = 128,
                        palm_class: int = 1,
                        method: str = 'watershed',
                        geometry: str = 'circle',
                        **extract_kwargs):
    """Run segmentation then extraction in one call. The heatmap is always
    written and used here for the most precise extraction.
    """
    seg = segment(model, raster_path, out_dir, tile_size=tile_size,
                  overlap=overlap, palm_class=palm_class, write_heatmap=True)
    stem = osp.splitext(osp.basename(raster_path))[0]
    out_vector_path = out_vector_path or osp.join(out_dir, f'{stem}_trees.gpkg')
    ext = extract(seg.label_path, out_vector_path, heatmap_path=seg.prob_path,
                  palm_class=palm_class, method=method, geometry=geometry,
                  **extract_kwargs)
    return seg, ext
