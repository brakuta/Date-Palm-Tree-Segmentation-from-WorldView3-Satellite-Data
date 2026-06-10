#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Segmentation Tiling Pipeline  —  Production Version
=============================================================
Tiles georeferenced mosaics with polygon annotations into fixed-size
image tiles (.tif) and corresponding single-band semantic mask tiles (.png).

Compatible with:
  MMSegmentation, SegFormer, Mask2Former (semantic mode),
  DeepLab, UNet, and any framework expecting image + uint8 mask pairs.

Supported annotation formats
-----------------------------
  .shp      Shapefile
  .gdb      File Geodatabase  (requires "layer" key per source)
  .gpkg     GeoPackage        (requires "layer" key if multi-layer)
  .geojson  GeoJSON

Class encoding
--------------
  Background (unannotated pixels) = 0   (always)
  User-defined classes             = 1, 2, 3, ...  (see CLASS_MAP)

  Class values are rasterised in ascending order; higher values are
  rasterised last and therefore take priority where polygons overlap.
  Define CLASS_MAP carefully for your application.

Tiling strategy
---------------
  - Tiles generated within each split region polygon, anchored to region
    bounding box top-left (snapped to raster pixel grid).
  - Regions clipped to mosaic bounding box before tiling.
  - BOUNDARY_MODE controls partial edge tile handling:
      "strict" → drop tiles not fully inside the region polygon
      "mask"   → keep edge tiles; blacken image pixels and set mask
                 pixels to 0 (background) outside the region boundary.
                 Annotation polygons outside the boundary are excluded.

Overlap policy
--------------
  OVERLAP applies to training tiles only (both X and Y).
  Val/test tiles always use stride = tile_size (no overlap).
  Note: for semantic segmentation, overlapping tiles are less critical
  than for instance segmentation — the same pixel gets the same label
  regardless of which tile it appears in. Use lower overlap than for
  instance segmentation.

Contrast stretching
-------------------
  When APPLY_CONTRAST_STRETCH = True, per-band p2/p98 stretch is
  computed ONCE from the mosaic overview and applied uniformly to all
  tiles. Stretch parameters are saved to tiling_log.json for inference.

═══════════════════════════════════════════════════════════════════
RESOLUTION CHANGE GUIDE
═══════════════════════════════════════════════════════════════════

  Parameter            UAV 5 cm   Aerial 15 cm   WV3 30 cm
  ─────────────────    ────────   ────────────   ─────────
  OVERLAP (px)         256        128            64
    Note: semantic seg needs less overlap than instance seg.
    Rule: target 15–30 m physical overlap.
    Formula: OVERLAP = round(target_overlap_m / GSD)

  MIN_VISIBLE_AREA_M2  0.25       0.5            1.0
    Polygons smaller than this are skipped during rasterisation.
    Scale with GSD: at 30cm, 1.0 m² = ~11 px² (minimum meaningful mask).

  TILE_SIZE (px)       1024       1024           1024
    Ground coverage: 5cm→51m, 15cm→154m, 30cm→307m.

  BANDS                None       None           [1,2,3]
    [1,2,3] to drop alpha channel from RGBA images.

  MIN_IMAGE_COVERAGE   0.5        0.5            0.5
    Tiles with > 50% black/nodata area are dropped.

  KEEP_BACKGROUND_TILES:
    True  = keep tiles that are 100% background (no annotation polygons).
            Useful for LULC where background = "bare ground" or "other".
    False = drop fully-background tiles (saves disk space for tree mapping
            in sparse landscapes).

═══════════════════════════════════════════════════════════════════

USAGE
-----
1. Edit CONFIGURATION section.
2. Run:  python semseg_tile_pipeline.py

OUTPUT
------
    out_dir/
        train/
            images/  <job>_train_tile_000000.tif
            masks/   <job>_train_tile_000000.png   (uint8, values = class IDs)
        val/
            images/  ...
            masks/   ...
        test/
            images/  ...
            masks/   ...
        tile_index.gpkg        (spatial index of all tiles)
        class_palette.json     (class name → pixel value mapping)
        tiling_log.json        (config, balance, stretch params, versions)

REQUIREMENTS
------------
    pip install rasterio geopandas shapely tqdm numpy Pillow

MMSegmentation config note
--------------------------
    In your MMSeg dataset config, set:
        reduce_zero_label = False   (0 is background, keep it)
        num_classes = len(CLASS_MAP) + 1  (+ 1 for background)
    Point img_dir and ann_dir to the images/ and masks/ folders.
"""

import json
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.strtree import STRtree
from tqdm import tqdm


# =============================================================================
# CONFIGURATION  —  edit this section only
# =============================================================================

OUTPUT_DIR = Path(r"H:\Regional Date Palm Mapping\SemSeg_Output")
TILE_SIZE  = 1024
OVERLAP    = 128              # training overlap in pixels (both X and Y)
                              # val/test tiles always non-overlapping
                              # semantic seg: use lower overlap than instance

# ── CLASS MAP ─────────────────────────────────────────────────────────────────
# Define every class and its integer mask value.
# Background (unannotated pixels) is always 0 — do not include it here.
# Classes are rasterised in ascending value order; higher values win overlaps.
#
# Examples for different applications:
#
#   Tree mapping (single class):
#     CLASS_MAP = {"Tree": 1}
#
#   Date palm mapping:
#     CLASS_MAP = {"DatePalm": 1}
#
#   LULC (multi-class):
#     CLASS_MAP = {
#         "Bare_Ground":  1,
#         "Vegetation":   2,
#         "Built_Up":     3,
#         "Water":        4,
#         "Agriculture":  5,
#     }
#
#   Tree species:
#     CLASS_MAP = {"Acacia": 1, "Ghaf": 2, "DatePalm": 3}

CLASS_MAP = {
    "DatePalm": 1,
}

# Contrast stretching
APPLY_CONTRAST_STRETCH = True
STRETCH_LOWER_PCT      = 2
STRETCH_UPPER_PCT      = 98

# Polygon filtering
MIN_VISIBLE_AREA_M2  = 0.5   # skip polygons smaller than this (m²)
                              # increase for coarser resolutions (≥ 15 cm)
SIMPLIFY_TOLERANCE_M = 0.0   # simplify polygon vertices (0 = off)

# Tile filtering
KEEP_BACKGROUND_TILES = False # False = drop tiles that are 100% background
                              # True  = keep all tiles (use for LULC where
                              #         background itself is a meaningful class)
DROP_NODATA_TILES     = True  # True  = skip entirely black/nodata tiles
MIN_IMAGE_COVERAGE    = 0.5   # minimum fraction of non-zero image pixels

# Boundary handling
# "strict" → drop tiles not fully inside the split region polygon
# "mask"   → keep edge tiles; blacken image + set mask to 0 outside boundary
BOUNDARY_MODE = "mask"

# Debugging
GENERATE_DEBUG_SHAPES = False # write clipped split boundaries to .gpkg

# Resume
SKIP_EXISTING = False         # True = skip tiles whose mask .png already exists

# Processing
MAX_WORKERS = 4
BANDS       = None            # None = all bands. [1,2,3] = RGB from RGBA.

# ── JOBS ─────────────────────────────────────────────────────────────────────
# Each job defines one mosaic to tile.
#
# Required keys:
#   name       : unique identifier, used as tile filename prefix
#   mosaic     : path to GeoTIFF
#   id_field   : "auto" (ignored for sem seg; kept for consistency)
#
# Annotation sources — list of dicts, one per class layer:
#   sources: [
#     {
#       "shapefile"  : path to annotation file
#       "layer"      : feature class name (required for .gdb / multi-layer .gpkg)
#       "class_name" : must match a key in CLASS_MAP
#       "class_field": (optional) attribute column that specifies the class.
#                      If provided, each polygon's class is read from this column.
#                      The column value must match a key in CLASS_MAP.
#                      Use this when one shapefile contains multiple classes.
#     },
#     ...
#   ]
#
# Split assignment — choose ONE per job:
#   split                         : "train" / "val" / "test"
#   split_shapefile + split_field : spatial region file

JOBS = [

    # ── Date palm semantic segmentation ──────────────────────────────────────
    {
        "name":   "WV3_Ajman",
        "mosaic": Path(r"Z:\Final Geodatabase\National Date Palm Mapping\Subset Multiscale Data\Wordview3\Ajman_WV3_2021_30cm.tif"),
        "sources": [
            {
                "shapefile":  Path(r"Z:\Final Geodatabase\National Date Palm Mapping\Vector Data\Datepalm.gdb"),
                "layer":      "WV3_Datepalm__Ajman",
                "class_name": "DatePalm",
            },
        ],
        "id_field":        "auto",
        "split_shapefile": Path(r"Z:\Final Geodatabase\National Date Palm Mapping\Subset Multiscale Data\Area_Datepalm\Datepalm_WV3_Area.shp"),
        "split_field":     "Task",
        "overlap":         64,
        "bands":           [1, 2, 3],
    },

    # ── LULC multi-class example (uncomment and adapt) ────────────────────────
    # {
    #     "name":   "LULC_Ajman",
    #     "mosaic": Path(r"Z:\...\Ajman_mosaic.tif"),
    #     "sources": [
    #         {"shapefile": Path(r"Z:\...\lulc.gdb"), "layer": "Bare_Ground",  "class_name": "Bare_Ground"},
    #         {"shapefile": Path(r"Z:\...\lulc.gdb"), "layer": "Vegetation",   "class_name": "Vegetation"},
    #         {"shapefile": Path(r"Z:\...\lulc.gdb"), "layer": "Built_Up",     "class_name": "Built_Up"},
    #         {"shapefile": Path(r"Z:\...\lulc.gdb"), "layer": "Water",        "class_name": "Water"},
    #     ],
    #     "id_field":        "auto",
    #     "split_shapefile": Path(r"Z:\...\splits.shp"),
    #     "split_field":     "split",
    #     "overlap":         128,
    #     "bands":           None,
    # },

    # ── Single shapefile with class_field (one file, multiple classes) ────────
    # {
    #     "name":   "Trees_Fujairah",
    #     "mosaic": Path(r"Z:\...\Fujairah_mosaic.tif"),
    #     "sources": [
    #         {
    #             "shapefile":   Path(r"Z:\...\trees.shp"),
    #             "class_field": "species",   # column values must be in CLASS_MAP
    #         },
    #     ],
    #     "id_field":        "auto",
    #     "split_shapefile": Path(r"Z:\...\splits.shp"),
    #     "split_field":     "split",
    #     "overlap":         128,
    #     "bands":           [1, 2, 3],
    # },
]

# =============================================================================
# END OF CONFIGURATION  —  no changes needed below this line
# =============================================================================

VALID_SPLITS = {"train", "val", "test"}


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================

def preflight_checks(jobs: List[Dict]) -> None:
    """Validate all inputs before any processing begins."""
    errors = []

    # Validate CLASS_MAP
    if not CLASS_MAP:
        errors.append("  CLASS_MAP is empty. Define at least one class.")
    if 0 in CLASS_MAP.values():
        errors.append(
            "  CLASS_MAP contains value 0. Background is always 0 — "
            "use values >= 1 for all classes."
        )
    if len(set(CLASS_MAP.values())) != len(CLASS_MAP):
        errors.append("  CLASS_MAP contains duplicate class values.")

    # Duplicate job names
    names = [j["name"] for j in jobs]
    seen  = set()
    for n in names:
        if n in seen:
            errors.append(f"  Duplicate job name: '{n}'.")
        seen.add(n)

    for job in jobs:
        name = job.get("name", "<unnamed>")

        # Mosaic
        mosaic = Path(job.get("mosaic", ""))
        if not mosaic.exists():
            errors.append(f"  [{name}] Mosaic not found:\n       {mosaic}")
        else:
            # Dtype check
            try:
                with rasterio.open(str(mosaic)) as _src:
                    dtype = _src.dtypes[0]
                    if dtype != "uint8":
                        errors.append(
                            f"  [{name}] Mosaic dtype is '{dtype}'. "
                            f"Expected uint8. Convert with:\n"
                            f"       gdal_translate -ot Byte -scale "
                            f"{mosaic.name} {mosaic.stem}_8bit.tif"
                        )
            except Exception as e:
                errors.append(f"  [{name}] Cannot open mosaic: {e}")

        # Annotation sources
        sources = job.get("sources", [])
        if not sources:
            errors.append(f"  [{name}] No annotation sources defined.")
        for src_def in sources:
            shp = Path(src_def.get("shapefile", ""))
            if not shp.exists():
                errors.append(f"  [{name}] Source shapefile not found:\n"
                              f"       {shp}")
            if shp.suffix.lower() == ".gdb" and "layer" not in src_def:
                errors.append(
                    f"  [{name}] GDB source requires a 'layer' key: {shp.name}"
                )
            # Validate class names
            if "class_field" not in src_def:
                cname = src_def.get("class_name", "")
                if cname not in CLASS_MAP:
                    errors.append(
                        f"  [{name}] class_name '{cname}' not in CLASS_MAP. "
                        f"Add it to CLASS_MAP or fix the spelling."
                    )

        # Split assignment
        if "split" not in job:
            split_shp = Path(job.get("split_shapefile", ""))
            if not split_shp.exists():
                errors.append(
                    f"  [{name}] Split shapefile not found:\n       {split_shp}"
                )
            if "split_field" not in job:
                errors.append(f"  [{name}] Missing 'split_field' key.")
        else:
            if job["split"] not in VALID_SPLITS:
                errors.append(
                    f"  [{name}] Invalid split '{job['split']}'. "
                    f"Must be one of: {VALID_SPLITS}"
                )

        # Overlap
        job_overlap = job.get("overlap", OVERLAP)
        if not (0 <= job_overlap < TILE_SIZE):
            errors.append(
                f"  [{name}] overlap={job_overlap} must be "
                f">= 0 and < {TILE_SIZE}."
            )

    if errors:
        raise ValueError(
            "\n\nPre-flight validation failed:\n\n"
            + "\n\n".join(errors)
            + "\n\nFix the above before running."
        )

    print(f"  Pre-flight checks passed for {len(jobs)} job(s).")
    print(f"  CLASS_MAP: {CLASS_MAP}")


# =============================================================================
# VECTOR DATA LOADING
# =============================================================================

def read_vector(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """Read .shp / .gdb / .gpkg / .geojson into a GeoDataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".gdb":
        return gpd.read_file(str(path), layer=layer)
    if suffix == ".gpkg" and layer:
        return gpd.read_file(str(path), layer=layer)
    return gpd.read_file(str(path))


def load_annotation_sources(
    sources: List[Dict],
    raster_crs,
) -> List[Tuple[int, object]]:
    """
    Load all annotation sources for one job.

    Returns a list of (class_value, shapely_geometry) pairs, sorted by
    class_value ascending so that higher-value classes are rasterised last
    and take priority in overlapping regions.

    Handles:
      - Single-class sources (class_name key)
      - Multi-class sources (class_field key — reads class from attribute)
    """
    all_polygons: List[Tuple[int, object]] = []

    for src_def in sources:
        shp_path    = Path(src_def["shapefile"])
        layer       = src_def.get("layer")
        class_name  = src_def.get("class_name")
        class_field = src_def.get("class_field")

        gdf = read_vector(shp_path, layer=layer)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        gdf = gdf[
            gdf.geometry.notna() & ~gdf.geometry.is_empty
        ].reset_index(drop=True)

        for _, row in gdf.iterrows():
            geom = row.geometry
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            if geom.area < MIN_VISIBLE_AREA_M2:
                continue

            if class_field is not None:
                # Class determined per-polygon from attribute column
                cname = str(row[class_field]).strip()
                if cname not in CLASS_MAP:
                    continue   # silently skip unknown classes
                cval = CLASS_MAP[cname]
            else:
                cval = CLASS_MAP[class_name]

            if SIMPLIFY_TOLERANCE_M > 0:
                geom = geom.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
                if geom.is_empty:
                    continue

            all_polygons.append((cval, geom))

    # Sort by class value ascending — higher values rasterised last = win overlaps
    all_polygons.sort(key=lambda x: x[0])
    return all_polygons


# =============================================================================
# MASK RASTERISATION
# =============================================================================

def rasterise_mask(
    polygons: List[Tuple[int, object]],
    tile_transform,
    tile_size: int,
    region_geom=None,
    boundary_mode: str = "strict",
) -> np.ndarray:
    """
    Rasterise annotation polygons into a uint8 semantic mask.

    Args:
        polygons       : list of (class_value, shapely_geometry) pairs
        tile_transform : rasterio Affine transform of the tile
        tile_size      : tile width and height in pixels
        region_geom    : split region polygon (used in mask mode to
                         set outside pixels to 0)
        boundary_mode  : "strict" or "mask"

    Returns:
        (tile_size, tile_size) uint8 array.
        Pixel value = class ID. Background = 0.
    """
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    if not polygons:
        return mask

    # Determine the annotation boundary
    if boundary_mode == "mask" and region_geom is not None:
        # Build the tile box and intersect with the region
        from shapely.geometry import box as _box
        x0 = tile_transform.c
        y1 = tile_transform.f
        x1 = x0 + tile_size * tile_transform.a
        y0 = y1 + tile_size * tile_transform.e
        tile_box     = _box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        annotation_geom = tile_box.intersection(region_geom)
        if annotation_geom.is_empty:
            annotation_geom = None
    else:
        annotation_geom = None

    # Rasterise each class in ascending order (higher classes win overlaps)
    for class_val, geom in polygons:
        # Clip polygon to annotation boundary
        if annotation_geom is not None:
            try:
                clipped = geom.intersection(annotation_geom)
            except Exception:
                clipped = geom
        else:
            clipped = geom

        if clipped is None or clipped.is_empty:
            continue

        # Flatten MultiPolygon / GeometryCollection to simple geometries
        if isinstance(clipped, GeometryCollection):
            parts = [
                g for g in clipped.geoms
                if isinstance(g, (Polygon, MultiPolygon))
            ]
            if not parts:
                continue
        else:
            parts = [clipped]

        for part in parts:
            if isinstance(part, MultiPolygon):
                sub_parts = list(part.geoms)
            else:
                sub_parts = [part]
            for sub in sub_parts:
                if sub.is_empty or sub.area <= 0:
                    continue
                burned = rasterize(
                    [(sub, class_val)],
                    out_shape=(tile_size, tile_size),
                    transform=tile_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                )
                mask[burned == class_val] = class_val

    # In mask mode: enforce that pixels outside the region boundary stay 0
    if boundary_mode == "mask" and region_geom is not None:
        from rasterio.features import rasterize as _rast
        region_mask = _rast(
            [(region_geom, 1)],
            out_shape=(tile_size, tile_size),
            transform=tile_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
        mask[region_mask == 0] = 0

    return mask


# =============================================================================
# SPLIT REGION HANDLING
# =============================================================================

def load_split_regions(
    shp_path: Path,
    split_field: str,
    split_layer: Optional[str],
    target_crs,
) -> gpd.GeoDataFrame:
    """Load, validate, and reproject a split region file."""
    gdf = read_vector(shp_path, layer=split_layer)
    if split_field not in gdf.columns:
        raise KeyError(
            f"Column '{split_field}' not found in {shp_path.name}.\n"
            f"Available columns: {list(gdf.columns)}"
        )
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf[split_field] = gdf[split_field].astype(str).str.strip().str.lower()
    invalid = set(gdf[split_field].unique()) - VALID_SPLITS
    if invalid:
        raise ValueError(
            f"Split field '{split_field}' contains invalid values: {invalid}.\n"
            f"Allowed values: {VALID_SPLITS}.\n"
            f"Values in file: {list(gdf[split_field].unique())}"
        )
    return gdf


def is_tile_in_region(
    tile_geom,
    region_tree: STRtree,
    region_geoms: np.ndarray,
    region_idx: int,
) -> bool:
    """Return False if the tile genuinely overlaps a different region."""
    tile_area  = tile_geom.area
    candidates = region_tree.query(tile_geom)
    for i in candidates:
        if i == region_idx:
            continue
        inter = region_geoms[i].intersection(tile_geom)
        if not inter.is_empty and inter.area > tile_area * 1e-6:
            return False
    return True


# =============================================================================
# TILE GRID
# =============================================================================

def compute_region_tile_grid(
    region_geom,
    raster_transform,
    raster_width: int,
    raster_height: int,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Compute (row, col, x_off, y_off) for all tile positions covering one
    split region. Grid anchored to region bounding box top-left corner,
    snapped to the nearest raster pixel.
    """
    stride = tile_size - overlap
    minx, miny, maxx, maxy = region_geom.bounds
    inv       = ~raster_transform
    col_start = max(0, int(math.floor((inv * (minx, maxy))[0])))
    row_start = max(0, int(math.floor((inv * (minx, maxy))[1])))
    col_end   = min(raster_width,  int(math.ceil((inv * (maxx, miny))[0])))
    row_end   = min(raster_height, int(math.ceil((inv * (maxx, miny))[1])))

    region_w = col_end - col_start
    region_h = row_end - row_start
    if region_w <= 0 or region_h <= 0:
        return []

    n_cols = math.ceil((region_w - overlap) / stride) if region_w > overlap else 1
    n_rows = math.ceil((region_h - overlap) / stride) if region_h > overlap else 1

    return [
        (r, c, col_start + c * stride, row_start + r * stride)
        for r in range(n_rows)
        for c in range(n_cols)
        if col_start + c * stride < raster_width
        and row_start + r * stride < raster_height
    ]


# =============================================================================
# CONTRAST STRETCHING
# =============================================================================

def compute_stretch_params(
    src: rasterio.io.DatasetReader,
    bands: Optional[List[int]],
    lower_pct: float,
    upper_pct: float,
) -> List[Tuple[float, float]]:
    """
    Compute per-band (p_low, p_high) from a mosaic overview.
    Called ONCE per job; parameters applied uniformly to all tiles.
    """
    ovr_factor = max(1, min(src.width, src.height) // 2048)
    out_w      = max(1, src.width  // ovr_factor)
    out_h      = max(1, src.height // ovr_factor)
    n_bands    = len(bands) if bands else src.count

    overview = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=Resampling.nearest,
    )
    if bands:
        overview = overview[[b - 1 for b in bands], :, :]

    params = []
    for i in range(n_bands):
        band  = overview[i].astype(np.float32)
        valid = band[band > 0]
        if len(valid) == 0:
            params.append((0.0, 255.0))
            continue
        p_low  = float(np.percentile(valid, lower_pct))
        p_high = float(np.percentile(valid, upper_pct))
        if p_high - p_low < 1:
            p_high = p_low + 1
        params.append((p_low, p_high))
    return params


def apply_stretch(
    img_data: np.ndarray,
    stretch_params: List[Tuple[float, float]],
) -> np.ndarray:
    """Apply pre-computed stretch parameters to a tile. Returns uint8."""
    out = np.zeros_like(img_data, dtype=np.uint8)
    for i, (p_low, p_high) in enumerate(stretch_params):
        band      = img_data[i].astype(np.float32)
        zero_mask = (band == 0)
        stretched = np.clip((band - p_low) / (p_high - p_low) * 255.0, 0, 255)
        stretched[zero_mask] = 0
        out[i] = stretched.astype(np.uint8)
    return out


# =============================================================================
# IMAGE READ
# =============================================================================

def read_tile(
    src: rasterio.io.DatasetReader,
    x_off: int,
    y_off: int,
    tile_size: int,
    bands: Optional[List[int]],
) -> Tuple[np.ndarray, bool]:
    """Read one tile, select bands, zero-pad at mosaic edges."""
    win_w = min(tile_size, src.width  - x_off)
    win_h = min(tile_size, src.height - y_off)
    n_out = len(bands) if bands else src.count

    if win_w <= 0 or win_h <= 0:
        return np.zeros((n_out, tile_size, tile_size), dtype=src.dtypes[0]), False

    data = src.read(window=Window(x_off, y_off, win_w, win_h))
    if bands:
        data = data[[b - 1 for b in bands], :, :]

    if win_w < tile_size or win_h < tile_size:
        full = np.zeros((data.shape[0], tile_size, tile_size), dtype=data.dtype)
        full[:, :win_h, :win_w] = data
        data = full

    nodata   = src.nodata
    has_data = (
        bool((data != nodata).any()) if nodata is not None
        else bool((data != 0).any())
    )
    return data, has_data


# =============================================================================
# TILE WRITER
# =============================================================================

def write_tile_pair(
    img_path: Path,
    mask_path: Path,
    img_data: np.ndarray,
    mask_data: np.ndarray,
    tile_transform,
    raster_crs,
    src_profile: Dict,
    tile_size: int,
    n_bands: int,
) -> None:
    """Write image tile (.tif) and semantic mask (.png)."""
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # Image: compressed GeoTIFF
    profile = {
        **src_profile,
        "height":     tile_size,
        "width":      tile_size,
        "count":      n_bands,
        "transform":  tile_transform,
        "crs":        raster_crs,
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
        "dtype":      img_data.dtype,
    }
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(img_data)

    # Mask: single-band uint8 PNG (standard for semantic segmentation)
    # PNG preserves exact integer class values without lossy compression.
    Image.fromarray(mask_data, mode="L").save(mask_path)


# =============================================================================
# POST-TILING CHECKS
# =============================================================================

def integrity_check(output_dir: Path) -> List[str]:
    """Check for orphaned files and spatial overlap between splits."""
    warnings_out = []

    for split in VALID_SPLITS:
        img_dir  = output_dir / split / "images"
        mask_dir = output_dir / split / "masks"
        if not img_dir.exists():
            continue

        tifs  = {p.stem for p in img_dir.glob("*.tif")}
        masks = {p.stem for p in mask_dir.glob("*.png")} if mask_dir.exists() else set()

        orphan_img  = tifs  - masks
        orphan_mask = masks - tifs
        if orphan_img:
            warnings_out.append(
                f"[{split}] {len(orphan_img)} image(s) with no matching mask."
            )
        if orphan_mask:
            warnings_out.append(
                f"[{split}] {len(orphan_mask)} mask(s) with no matching image."
            )

    # Spatial overlap between splits
    idx_path = output_dir / "tile_index.gpkg"
    if idx_path.exists():
        try:
            idx = gpd.read_file(idx_path)
            for s1, s2 in [("train","val"), ("train","test"), ("val","test")]:
                g1 = idx[idx["split"] == s1]
                g2 = idx[idx["split"] == s2]
                if g1.empty or g2.empty:
                    continue
                joined = gpd.sjoin(
                    g1[["geometry"]], g2[["geometry"]],
                    how="inner", predicate="intersects"
                )
                if len(joined) > 0:
                    warnings_out.append(
                        f"Spatial overlap between {s1} and {s2}: "
                        f"{len(joined)} tile pair(s). Check split shapefile."
                    )
        except Exception as e:
            warnings_out.append(f"Spatial overlap check failed: {e}")

    return warnings_out


def class_pixel_balance(output_dir: Path) -> Dict:
    """
    Count pixels per class per split from the generated masks.
    More meaningful than tile counts for semantic segmentation.
    """
    report = {}
    inv_map = {v: k for k, v in CLASS_MAP.items()}
    inv_map[0] = "background"

    for split in VALID_SPLITS:
        mask_dir = output_dir / split / "masks"
        if not mask_dir.exists():
            report[split] = {}
            continue

        pixel_counts: Dict[int, int] = {}
        n_tiles     = 0
        bg_only     = 0

        for mp in mask_dir.glob("*.png"):
            mask = np.array(Image.open(mp))
            n_tiles += 1
            unique, counts = np.unique(mask, return_counts=True)
            has_fg = False
            for u, c in zip(unique, counts):
                pixel_counts[int(u)] = pixel_counts.get(int(u), 0) + int(c)
                if u != 0:
                    has_fg = True
            if not has_fg:
                bg_only += 1

        total_px = sum(pixel_counts.values()) or 1
        class_report = {}
        for cval, cname in sorted(inv_map.items()):
            px = pixel_counts.get(cval, 0)
            class_report[cname] = {
                "class_value":  cval,
                "pixel_count":  px,
                "pixel_pct":    round(100.0 * px / total_px, 2),
            }

        report[split] = {
            "tiles":              n_tiles,
            "background_only_tiles": bg_only,
            "class_pixels":       class_report,
        }

    return report


def reproducibility_record() -> Dict:
    """Capture library versions for publication reproducibility."""
    import sys as _sys
    import platform as _plat
    record = {"python": _sys.version, "platform": _plat.platform()}
    for pkg in ("rasterio", "geopandas", "shapely", "numpy", "Pillow", "fiona"):
        try:
            mod = __import__(pkg if pkg != "Pillow" else "PIL")
            record[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            record[pkg] = "not installed"
    return record


# =============================================================================
# SINGLE JOB PROCESSING
# =============================================================================

def process_job(job: Dict, output_dir: Path) -> Dict:
    """
    Tile one mosaic and generate image + semantic mask pairs.

    Processing order per tile:
      1. Boundary check (strict or mask mode)
      2. Straddling check (split data leakage prevention)
      3. Read raw image pixels
      4. Apply image mask if BOUNDARY_MODE = "mask"
      5. Check DROP_NODATA_TILES
      6. Apply contrast stretch
      7. Check MIN_IMAGE_COVERAGE (on post-mask, pre-stretch data)
      8. Rasterise semantic mask (polygons clipped to annotation boundary)
      9. Check KEEP_BACKGROUND_TILES
     10. Write image tile + mask tile
    """
    job_name    = job["name"]
    job_overlap = job.get("overlap", OVERLAP)
    job_bands   = job.get("bands", BANDS)

    print(f"\n  +-- Job: {job_name}")

    # ── Open mosaic ──────────────────────────────────────────────────────────
    src         = rasterio.open(job["mosaic"])
    raster_crs  = src.crs
    raster_tfm  = src.transform
    src_profile = src.profile.copy()
    gsd         = abs(raster_tfm.a)
    n_out_bands = len(job_bands) if job_bands else src.count

    print(f"  |   mosaic    : {Path(job['mosaic']).name}")
    print(f"  |   size      : {src.width} x {src.height} px, "
          f"GSD={gsd:.4f} m, {src.count} band(s) -> {n_out_bands} out")
    print(f"  |   CRS       : {raster_crs}")

    # ── Stretch parameters ────────────────────────────────────────────────────
    stretch_params = None
    if APPLY_CONTRAST_STRETCH:
        print(f"  |   stretch   : computing p{STRETCH_LOWER_PCT}/"
              f"p{STRETCH_UPPER_PCT} from mosaic overview...")
        stretch_params = compute_stretch_params(
            src, job_bands, STRETCH_LOWER_PCT, STRETCH_UPPER_PCT
        )
        for i, (pl, ph) in enumerate(stretch_params):
            print(f"  |              band {i+1}: [{pl:.1f}, {ph:.1f}] -> [0, 255]")

    # ── Load annotation polygons ──────────────────────────────────────────────
    print(f"  |   loading   : {len(job['sources'])} annotation source(s)...")
    all_polygons = load_annotation_sources(job["sources"], raster_crs)
    print(f"  |   polygons  : {len(all_polygons)} total "
          f"({dict((k, sum(1 for v,_ in all_polygons if v==CLASS_MAP[k])) for k in CLASS_MAP)})")

    # Build spatial index for fast per-tile querying
    poly_geoms  = [g for _, g in all_polygons]
    poly_values = [v for v, _ in all_polygons]
    poly_tree   = STRtree(poly_geoms)

    # ── Split regions ─────────────────────────────────────────────────────────
    if "split" in job:
        fixed_split = job["split"]
        x0 = raster_tfm.c; y1 = raster_tfm.f
        x1 = x0 + src.width  * raster_tfm.a
        y0 = y1 + src.height * raster_tfm.e
        regions_gdf = gpd.GeoDataFrame(
            {"split": [fixed_split],
             "geometry": [box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))]},
            crs=raster_crs,
        )
        split_col = "split"
        print(f"  |   split     : fixed -> {fixed_split}")
    else:
        regions_gdf = load_split_regions(
            Path(job["split_shapefile"]),
            job["split_field"],
            job.get("split_layer"),
            raster_crs,
        )
        split_col = job["split_field"]
        print(f"  |   split     : region-based, "
              f"{dict(regions_gdf[split_col].value_counts())}")

    # Clip to mosaic bounding box
    x0 = raster_tfm.c; y1 = raster_tfm.f
    x1 = x0 + src.width  * raster_tfm.a
    y0 = y1 + src.height * raster_tfm.e
    mosaic_bbox = box(min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regions_gdf = regions_gdf.copy()
        regions_gdf["geometry"] = regions_gdf.geometry.intersection(mosaic_bbox)
    regions_gdf = regions_gdf[
        ~regions_gdf.geometry.is_empty
    ].reset_index(drop=True)

    if GENERATE_DEBUG_SHAPES:
        debug_path = output_dir / f"DEBUG_boundary_{job_name}.gpkg"
        regions_gdf.to_file(debug_path, driver="GPKG")
        print(f"  |   debug     : {debug_path.name}")

    region_geoms_arr  = regions_gdf.geometry.values
    region_labels_arr = regions_gdf[split_col].values
    region_tree       = STRtree(region_geoms_arr)

    # ── Per-region tile loop ──────────────────────────────────────────────────
    counts  = {s: {"tiles": 0, "bg_only_tiles": 0} for s in VALID_SPLITS}
    dropped = {
        "partial_edge": 0, "straddling": 0,
        "nodata": 0, "background_only": 0, "skipped_existing": 0,
    }
    tile_records = []

    def _max_existing_idx(split: str) -> int:
        img_dir = output_dir / split / "images"
        if not img_dir.exists():
            return 0
        prefix = f"{job_name}_{split}_tile_"
        indices = []
        for p in img_dir.glob(f"{prefix}*.tif"):
            try:
                indices.append(int(p.stem.replace(prefix, "")))
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0

    split_counters = {
        s: [_max_existing_idx(s) if SKIP_EXISTING else 0]
        for s in VALID_SPLITS
    }

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures  = []

    for reg_idx, (region_geom, region_split) in enumerate(
        zip(region_geoms_arr, region_labels_arr)
    ):
        reg_overlap = job_overlap if region_split == "train" else 0
        stride      = TILE_SIZE - reg_overlap
        grid = compute_region_tile_grid(
            region_geom, raster_tfm,
            src.width, src.height,
            TILE_SIZE, reg_overlap,
        )
        print(f"  |   region {reg_idx+1:>2}  [{region_split}]: "
              f"{len(grid)} positions  "
              f"(stride={stride} px, overlap={reg_overlap} px)")

        for row, col, x_off, y_off in tqdm(
            grid,
            desc=f"     {job_name} [{region_split}]",
            unit="tile", leave=False,
        ):
            tile_transform = src.window_transform(
                Window(x_off, y_off, TILE_SIZE, TILE_SIZE)
            )
            x0_w = tile_transform.c
            y1_w = tile_transform.f
            x1_w = x0_w + TILE_SIZE * tile_transform.a
            y0_w = y1_w + TILE_SIZE * tile_transform.e
            tile_geom = box(
                min(x0_w, x1_w), min(y0_w, y1_w),
                max(x0_w, x1_w), max(y0_w, y1_w),
            )

            # ── Boundary check ────────────────────────────────────────────────
            if BOUNDARY_MODE == "strict":
                if not region_geom.covers(tile_geom):
                    dropped["partial_edge"] += 1
                    continue
            elif BOUNDARY_MODE == "mask":
                if not region_geom.intersects(tile_geom):
                    continue

            # ── Straddling check ──────────────────────────────────────────────
            if not is_tile_in_region(
                tile_geom, region_tree, region_geoms_arr, reg_idx
            ):
                dropped["straddling"] += 1
                continue

            # ── Output paths ──────────────────────────────────────────────────
            split     = region_split
            idx       = split_counters[split][0]
            tile_name = f"{job_name}_{split}_tile_{idx:06d}"
            img_path  = output_dir / split / "images" / f"{tile_name}.tif"
            mask_path = output_dir / split / "masks"  / f"{tile_name}.png"

            if SKIP_EXISTING and mask_path.exists():
                dropped["skipped_existing"] += 1
                split_counters[split][0] += 1
                counts[split]["tiles"] += 1
                continue

            # ── Read raw image ────────────────────────────────────────────────
            img_data, has_data = read_tile(
                src, x_off, y_off, TILE_SIZE, job_bands
            )

            if DROP_NODATA_TILES and not has_data:
                dropped["nodata"] += 1
                continue

            # ── Apply image mask (BOUNDARY_MODE = "mask") ─────────────────────
            if BOUNDARY_MODE == "mask":
                from rasterio.features import rasterize as _rast
                region_mask_arr = _rast(
                    [(region_geom, 1)],
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    transform=tile_transform,
                    fill=0, dtype=np.uint8, all_touched=False,
                )
                img_data = img_data * region_mask_arr[np.newaxis, :, :]

            # ── Image coverage check (pre-stretch) ────────────────────────────
            if MIN_IMAGE_COVERAGE > 0.0:
                nodata_val = src.nodata
                valid_px   = (
                    int(np.count_nonzero(img_data != nodata_val))
                    if nodata_val is not None
                    else int(np.count_nonzero(img_data))
                )
                if valid_px / float(img_data.size) < MIN_IMAGE_COVERAGE:
                    dropped["nodata"] += 1
                    continue

            # ── Apply contrast stretch ────────────────────────────────────────
            if APPLY_CONTRAST_STRETCH and stretch_params is not None:
                img_data = apply_stretch(img_data, stretch_params)

            # ── Rasterise semantic mask ───────────────────────────────────────
            # Query only polygons that intersect this tile
            candidate_idxs = poly_tree.query(tile_geom)
            tile_polygons  = [
                (poly_values[i], poly_geoms[i])
                for i in candidate_idxs
                if poly_geoms[i].intersects(tile_geom)
            ]

            mask_data = rasterise_mask(
                tile_polygons,
                tile_transform,
                TILE_SIZE,
                region_geom=region_geom,
                boundary_mode=BOUNDARY_MODE,
            )

            # ── Background-only tile check ────────────────────────────────────
            has_foreground = bool(np.any(mask_data > 0))
            if not KEEP_BACKGROUND_TILES and not has_foreground:
                dropped["background_only"] += 1
                continue

            # ── Queue write ───────────────────────────────────────────────────
            future = executor.submit(
                write_tile_pair,
                img_path, mask_path,
                img_data, mask_data,
                tile_transform, raster_crs, src_profile,
                TILE_SIZE, n_out_bands,
            )
            futures.append((
                future, tile_name, split,
                has_foreground, tile_geom, idx,
            ))
            counts[split]["tiles"] += 1
            if not has_foreground:
                counts[split]["bg_only_tiles"] += 1
            split_counters[split][0] += 1

    # Wait for all writes
    for future, tile_name, split, has_fg, tile_geom, idx in futures:
        try:
            future.result()
            tile_records.append({
                "job_name":      job_name,
                "split":         split,
                "tile_id":       idx,
                "tile_name":     tile_name,
                "has_foreground": has_fg,
                "geometry":      tile_geom,
            })
        except Exception as e:
            print(f"\n  ✖  Write failed for {tile_name}: {e}")

    executor.shutdown(wait=True)
    src.close()

    # Per-job summary
    print(f"\n     Results for '{job_name}':")
    for s in ("train", "val", "test"):
        if counts[s]["tiles"] > 0:
            note = f"  (overlap={job_overlap} px)" if s == "train" and job_overlap > 0 else ""
            bg   = counts[s]["bg_only_tiles"]
            print(f"       {s:5s} : {counts[s]['tiles']:>5} tiles  "
                  f"({bg} background-only){note}")
    drop_msgs = [
        ("partial_edge",     "partial edge"),
        ("straddling",       "straddling boundary"),
        ("nodata",           "nodata/coverage"),
        ("background_only",  "background only"),
        ("skipped_existing", "already existed"),
    ]
    for key, label in drop_msgs:
        if dropped[key] > 0:
            print(f"       Dropped ({label:25s}): {dropped[key]}")

    return {
        "job_name":     job_name,
        "counts":       counts,
        "dropped":      dropped,
        "stretch_params": [
            {"band": i+1, "p_low": pl, "p_high": ph}
            for i, (pl, ph) in enumerate(stretch_params or [])
        ],
        "tile_records": tile_records,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()

    print(f"\n{'='*60}")
    print("  Semantic Segmentation Tiling Pipeline")
    print(f"{'='*60}")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"  Tile size    : {TILE_SIZE} x {TILE_SIZE} px")
    print(f"  Overlap      : {OVERLAP} px  (training only)")
    print(f"  Boundary     : {BOUNDARY_MODE}")
    print(f"  Stretch      : {APPLY_CONTRAST_STRETCH}")
    print(f"  Min coverage : {MIN_IMAGE_COVERAGE}")
    print(f"  Classes      : {CLASS_MAP}")
    print(f"  Workers      : {MAX_WORKERS}")
    print(f"  Jobs         : {len(JOBS)}")
    print(f"{'='*60}")

    print("\n  Running pre-flight checks...")
    preflight_checks(JOBS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save class palette for reference
    palette = {"background": 0, **CLASS_MAP}
    (OUTPUT_DIR / "class_palette.json").write_text(
        json.dumps(palette, indent=2)
    )
    print("  Class palette saved -> class_palette.json")

    all_records  = []
    job_logs     = []
    crs_registry = {}

    for job in JOBS:
        result = process_job(job, OUTPUT_DIR)
        all_records.extend(result.pop("tile_records"))
        job_logs.append(result)
        with rasterio.open(job["mosaic"]) as s:
            crs_registry[job["name"]] = str(s.crs)

    # Tile index
    if all_records:
        first_crs = list(crs_registry.values())[0]
        mixed_crs = len(set(crs_registry.values())) > 1
        idx       = gpd.GeoDataFrame(all_records, crs=first_crs)
        if mixed_crs:
            print(f"\n  ⚠  Multiple CRSes — reprojecting tile index to {first_crs}")
            for jn, jcrs in crs_registry.items():
                if jcrs == first_crs:
                    continue
                mask_ = idx["job_name"] == jn
                sub   = gpd.GeoDataFrame(idx[mask_], crs=jcrs, geometry="geometry")
                idx.loc[mask_, "geometry"] = sub.to_crs(first_crs).geometry.values
        idx.to_file(OUTPUT_DIR / "tile_index.gpkg", driver="GPKG")

    # Integrity check
    print("\n  Running post-tiling integrity checks...")
    issues = integrity_check(OUTPUT_DIR)
    if issues:
        print(f"  ⚠  {len(issues)} issue(s):")
        for iss in issues:
            print(f"       - {iss}")
    else:
        print("  ✓  All integrity checks passed.")

    # Class pixel balance
    balance = class_pixel_balance(OUTPUT_DIR)
    repro   = reproducibility_record()

    # Audit log
    elapsed = round(time.time() - t0, 1)
    log = {
        "pipeline_config": {
            "tile_size":               TILE_SIZE,
            "overlap_train_only":      OVERLAP,
            "boundary_mode":           BOUNDARY_MODE,
            "apply_contrast_stretch":  APPLY_CONTRAST_STRETCH,
            "stretch_lower_pct":       STRETCH_LOWER_PCT,
            "stretch_upper_pct":       STRETCH_UPPER_PCT,
            "min_visible_area_m2":     MIN_VISIBLE_AREA_M2,
            "keep_background_tiles":   KEEP_BACKGROUND_TILES,
            "drop_nodata_tiles":       DROP_NODATA_TILES,
            "min_image_coverage":      MIN_IMAGE_COVERAGE,
            "skip_existing":           SKIP_EXISTING,
            "max_workers":             MAX_WORKERS,
            "class_map":               CLASS_MAP,
        },
        "elapsed_seconds":   elapsed,
        "class_pixel_balance": balance,
        "reproducibility":   repro,
        "job_summaries":     job_logs,
        "crs_per_job":       crs_registry,
    }
    (OUTPUT_DIR / "tiling_log.json").write_text(json.dumps(log, indent=2))

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Finished in {elapsed}s")
    print(f"{'='*60}")

    print("\n  Class pixel balance:")
    for split in ("train", "val", "test"):
        b = balance.get(split, {})
        if not b:
            continue
        print(f"\n     [{split}]  {b['tiles']} tiles  "
              f"({b['background_only_tiles']} background-only)")
        for cname, cinfo in b.get("class_pixels", {}).items():
            print(f"       {cname:<20s}: {cinfo['pixel_count']:>12,} px  "
                  f"({cinfo['pixel_pct']:>6.2f}%)")

    print(f"\n     Output       -> {OUTPUT_DIR}")
    print("     Tile index   -> tile_index.gpkg")
    print("     Class palette-> class_palette.json")
    print("     Audit log    -> tiling_log.json")
    print()


if __name__ == "__main__":
    main()
