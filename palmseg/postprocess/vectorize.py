# Copyright (c) 2024. Licensed under Apache-2.0.
"""Vectorisation of palm instances to georeferenced polygons.

Converts tree instances (or a label raster) to polygons in the raster CRS and
writes GeoPackage/GeoJSON. ``instances_to_gdf`` vectorises the TreeInstance
objects from ``individual_trees`` and preserves per-tree attributes;
``mask_to_gdf`` vectorises a label raster directly with ``rasterio.features``.
``deduplicate_overlaps`` removes duplicate crowns by IoU using an STRtree index.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

import geopandas as gpd
import rasterio
from rasterio import features as rio_features
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, Point, shape as shapely_shape
from shapely.validation import make_valid

from palmseg.postprocess.individual_trees import TreeInstance


def _rio_to_shapely_matrix(transform) -> list:
    """rasterio Affine (a,b,c,d,e,f) -> shapely affine [a,b,d,e,c,f]."""
    a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    return [a, b, d, e, c, f]


def _largest_polygon(geom):
    from shapely.geometry.base import BaseMultipartGeometry
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    if isinstance(geom, BaseMultipartGeometry):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and g.area > 0]
        return max(polys, key=lambda g: g.area) if polys else None
    return None


def _contour_to_polygon_px(cnt: np.ndarray) -> Optional[Polygon]:
    pts = cnt[:, 0, :].astype(np.float64)        # (x=col, y=row)
    if pts.shape[0] < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = _largest_polygon(make_valid(poly))
    return poly if (poly is not None and poly.area > 0) else None


def _circle_px(centroid_rc, area_px, quad_segs=16) -> Optional[Polygon]:
    cy, cx = centroid_rc                          # row, col
    r = float(np.sqrt(max(area_px, 1.0) / np.pi))
    circ = Point(cx, cy).buffer(r, quad_segs=quad_segs)
    return circ if (circ.is_valid and circ.area > 0) else None


def instances_to_gdf(trees: List[TreeInstance],
                     transform,
                     crs,
                     geometry: str = 'circle',
                     quad_segs: int = 16,
                     simplify_tol_px: float = 0.5,
                     pixel_size_m: Optional[float] = None) -> gpd.GeoDataFrame:
    """Build a georeferenced GeoDataFrame from TreeInstance objects.

    Args:
        trees: list of TreeInstance (pixel coordinates).
        transform: rasterio Affine of the source raster.
        crs: source CRS.
        geometry: 'circle' (area-preserving), 'polygon' (simplified outline).
        quad_segs: circle smoothness.
        simplify_tol_px: simplify tolerance for 'polygon' mode (pixels).
        pixel_size_m: linear pixel size (m) for diameter reporting; if None it
            is derived from the transform.

    Returns:
        GeoDataFrame with columns: id, score, circularity, diam_m, area_px.
    """
    if pixel_size_m is None:
        pixel_size_m = float(np.sqrt(abs(transform.a * transform.e)))
    mat = _rio_to_shapely_matrix(transform)

    geoms, ids, scores, circs, diams, areas = [], [], [], [], [], []
    for i, t in enumerate(trees):
        if geometry == 'circle':
            poly_px = _circle_px(t.centroid_rc, t.area_px, quad_segs)
        else:
            poly_px = _contour_to_polygon_px(t.contour)
            if poly_px is not None and simplify_tol_px > 0:
                poly_px = poly_px.simplify(simplify_tol_px,
                                           preserve_topology=True)
                poly_px = _largest_polygon(poly_px) \
                    if not isinstance(poly_px, Polygon) else poly_px
        if poly_px is None or poly_px.is_empty:
            continue
        g = affine_transform(poly_px, mat)
        if g.is_empty or not g.is_valid:
            g = _largest_polygon(make_valid(g))
            if g is None:
                continue
        geoms.append(g)
        ids.append(i)
        scores.append(round(float(t.confidence), 4))
        circs.append(round(float(t.circularity), 4))
        area_m2 = t.area_px * (pixel_size_m ** 2)
        diams.append(round(float(2.0 * np.sqrt(area_m2 / np.pi)), 3)
                     if area_m2 > 0 else 0.0)
        areas.append(round(float(t.area_px), 1))

    return gpd.GeoDataFrame(
        dict(id=ids, score=scores, circularity=circs, diam_m=diams,
             area_px=areas, geometry=geoms),
        crs=crs)


def mask_to_gdf(label_raster_path: str,
                class_value: int = 1,
                simplify_tol_px: float = 0.0) -> gpd.GeoDataFrame:
    """Vectorise a label raster directly with rasterio.features.shapes.

    Interiors/holes are handled natively, so there is no O(n^2) inner-polygon
    subtraction. Returns one polygon per connected component of ``class_value``.
    """
    with rasterio.open(label_raster_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs
    target = (band == class_value).astype(np.uint8)
    geoms = []
    for geom, val in rio_features.shapes(target, mask=target.astype(bool),
                                         transform=transform):
        if val != 1:
            continue
        poly = shapely_shape(geom)
        if simplify_tol_px > 0:
            poly = poly.simplify(simplify_tol_px, preserve_topology=True)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            geoms.append(poly)
    return gpd.GeoDataFrame(
        dict(id=range(len(geoms)), geometry=geoms), crs=crs)




def deduplicate_overlaps(gdf: 'gpd.GeoDataFrame',
                         iou_threshold: float = 0.5) -> 'gpd.GeoDataFrame':
    """Remove duplicate/overlapping crowns using an STRtree spatial index.

    Crowns whose intersection-over-union exceeds ``iou_threshold`` are treated
    as duplicates; the higher-``score`` crown is kept. This is O(n log n) via
    the spatial index rather than O(n^2) pairwise tests, so it scales to
    scene- and mosaic-level outputs.

    Args:
        gdf: GeoDataFrame from :func:`instances_to_gdf` (must have a 'score'
            column; if absent, larger area wins).
        iou_threshold: IoU above which two crowns are duplicates.

    Returns:
        Filtered GeoDataFrame (index reset).
    """
    from shapely.strtree import STRtree
    if len(gdf) <= 1:
        return gdf.reset_index(drop=True)
    geoms = list(gdf.geometry.values)
    if 'score' in gdf.columns:
        rank = {idx: r for r, idx in enumerate(
            gdf.sort_values('score', ascending=False).index)}
    else:
        rank = {idx: r for r, idx in enumerate(
            gdf.assign(_a=gdf.geometry.area)
               .sort_values('_a', ascending=False).index)}
    tree = STRtree(geoms)
    keep = []
    suppressed = set()
    # Process strongest first.
    for idx in sorted(gdf.index, key=lambda i: rank[i]):
        if idx in suppressed:
            continue
        keep.append(idx)
        g = gdf.geometry.loc[idx]
        for j in tree.query(g):
            jidx = gdf.index[j]
            if jidx == idx or jidx in suppressed:
                continue
            h = gdf.geometry.loc[jidx]
            inter = g.intersection(h).area
            if inter <= 0:
                continue
            union = g.area + h.area - inter
            if union > 0 and inter / union >= iou_threshold:
                # keep whichever has the better rank (already idx, the stronger)
                if rank[jidx] > rank[idx]:
                    suppressed.add(jidx)
    return gdf.loc[sorted(keep)].reset_index(drop=True)


def save_vector(gdf: gpd.GeoDataFrame, out_path: str) -> str:
    """Write GPKG (recommended) or GeoJSON based on extension."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    driver = 'GPKG' if out_path.lower().endswith('.gpkg') else 'GeoJSON'
    gdf.to_file(out_path, driver=driver)
    return out_path
