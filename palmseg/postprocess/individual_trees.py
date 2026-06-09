# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Individual tree extraction from a semantic palm mask.

Converts a binary or semantic palm mask into labelled tree instances with crown
polygons. Operates on NumPy arrays and has no dependency on the model or on
MMSegmentation, so it can run on segmentation produced by any source.

Pipeline:
    find_seeds                one seed per crown
    assign_instances          label palm pixels by nearest seed
    instances_to_geometries   per-instance contour, area, circularity, confidence

Seeds are taken from the probability surface when one is supplied, otherwise
from the distance transform of the mask. Instance assignment uses a seeded
watershed by default, or exact Euclidean nearest-seed ('voronoi').

Reference: Al-Ruzouq et al., Ecological Indicators 163 (2024), 112110.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError('opencv-python is required for postprocessing') from exc

from scipy import ndimage as ndi

logger = logging.getLogger('palmseg.postprocess')


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _as_bool_mask(palm_mask: np.ndarray) -> np.ndarray:
    if palm_mask is None:
        raise ValueError('palm_mask is required')
    arr = np.asarray(palm_mask)
    if arr.ndim != 2:
        raise ValueError(f'palm_mask must be 2-D, got shape {arr.shape}')
    return arr.astype(bool, copy=False)


def _check_prob(prob_map: Optional[np.ndarray],
                shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if prob_map is None:
        return None
    arr = np.asarray(prob_map)
    if arr.ndim != 2:
        raise ValueError(f'prob_map must be 2-D, got shape {arr.shape}')
    if arr.shape != shape:
        raise ValueError(
            f'prob_map shape {arr.shape} != palm_mask shape {shape}')
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Step 2 - seed (peak) detection
# ---------------------------------------------------------------------------
def detect_peaks(surface: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Boolean map of local maxima of ``surface`` (morphological peak step).

    A maximum filter marks plateau maxima; the zero background is eroded and an
    XOR removes the background border so only true peaks remain.

    Args:
        surface: 2-D float surface (probability or distance transform).
        connectivity: 1 (4-connected) or 2 (8-connected, paper default).
    """
    footprint = ndi.generate_binary_structure(2, connectivity)
    local_max = ndi.maximum_filter(surface, footprint=footprint) == surface
    background = surface == 0
    eroded_bg = ndi.binary_erosion(background, structure=footprint,
                                   border_value=1)
    return local_max ^ eroded_bg


def find_seeds(palm_mask: np.ndarray,
               prob_map: Optional[np.ndarray] = None,
               blur_ksize: int = 5,
               prob_threshold: float = 0.5,
               min_distance: int = 0) -> np.ndarray:
    """Return (N, 2) (row, col) seed coordinates, one per tree.

    Two regimes:
      * prob_map given  -> blur the probability surface and find peaks on it,
        keeping only peaks inside the palm mask (most precise).
      * prob_map None   -> find peaks on the Euclidean distance transform of the
        mask (crown centres as the points farthest from the background). Lets
        the extractor run on a binary mask alone.

    Args:
        palm_mask: 2-D boolean/0-1 palm mask.
        prob_map: optional 2-D palm-class probability ([0,1]) or logit surface.
        blur_ksize: Gaussian kernel side for the probability surface (odd).
        prob_threshold: zero out probabilities <= this before peak finding.
        min_distance: if > 0, enforce a minimum pixel spacing between seeds by
            suppressing weaker nearby peaks (non-max suppression on seeds).

    Returns:
        (N, 2) int64 array of (row, col).
    """
    mask = _as_bool_mask(palm_mask)
    prob = _check_prob(prob_map, mask.shape)

    if prob is not None:
        surf = prob.copy()
        if prob_threshold is not None:
            surf[surf <= prob_threshold] = 0.0
        if blur_ksize and blur_ksize > 1:
            k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            surf = cv2.GaussianBlur(surf, (k, k), 0)
        score = surf
    else:
        # Distance transform: peaks are crown centres. No probability needed.
        score = ndi.distance_transform_edt(mask).astype(np.float32)

    peaks = detect_peaks(score) & mask

    # Collapse each connected plateau of equal-height maxima to a single
    # centroid seed (a flat peak, e.g. the centre of a distance transform,
    # otherwise yields one seed per plateau pixel). Distinct crowns have
    # disconnected peaks and are unaffected.
    labelled, n = ndi.label(peaks, structure=ndi.generate_binary_structure(2, 2))
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
    centroids = ndi.center_of_mass(peaks, labelled, np.arange(1, n + 1))
    seeds = np.rint(np.asarray(centroids, dtype=np.float64)).astype(np.int64)
    seeds = seeds.reshape(-1, 2)
    # A centroid may fall outside the mask for a ring-shaped plateau; snap it
    # to the nearest peak pixel of its component.
    for i in range(seeds.shape[0]):
        r, c = seeds[i]
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]) \
                or not peaks[r, c]:
            comp = np.argwhere(labelled == (i + 1))
            if comp.size:
                d = np.abs(comp - seeds[i]).sum(axis=1)
                seeds[i] = comp[d.argmin()]

    if min_distance and seeds.shape[0] > 1:
        seeds = _suppress_close_seeds(seeds, score, min_distance)
    return seeds


def _suppress_close_seeds(seeds: np.ndarray, score: np.ndarray,
                          min_distance: int) -> np.ndarray:
    """Greedy non-max suppression: keep stronger seeds, drop close weaker ones."""
    from scipy.spatial import cKDTree
    strengths = score[seeds[:, 0], seeds[:, 1]]
    order = np.argsort(-strengths)
    kept: List[int] = []
    tree = cKDTree(seeds)
    suppressed = np.zeros(seeds.shape[0], dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(idx)
        neigh = tree.query_ball_point(seeds[idx], r=min_distance)
        for j in neigh:
            if j != idx:
                suppressed[j] = True
    return seeds[np.sort(kept)]


# ---------------------------------------------------------------------------
# Step 3 - clustering (assign palm pixels to nearest seed)
# ---------------------------------------------------------------------------
def assign_instances(palm_mask: np.ndarray,
                     seeds: np.ndarray,
                     method: str = 'watershed',
                     prob_map: Optional[np.ndarray] = None) -> np.ndarray:
    """Label palm pixels into per-tree instances given seeds.

    Args:
        palm_mask: 2-D boolean/0-1 mask.
        seeds: (N, 2) (row, col) seed coordinates.
        method: 'watershed' (geodesic, scalable; default) or 'voronoi'
            (exact Euclidean nearest-seed via cKDTree).
        prob_map: optional; for watershed, ``-prob`` is used as the flooding
            relief so basins follow crown confidence. Falls back to the
            negative distance transform when absent.

    Returns:
        int32 label image; 0 background, i in 1..N corresponds to seeds[i-1].
    """
    mask = _as_bool_mask(palm_mask)
    prob = _check_prob(prob_map, mask.shape)
    if seeds.shape[0] == 0:
        logger.warning('assign_instances: no seeds; returning empty labels')
        return np.zeros(mask.shape, dtype=np.int32)

    if method == 'voronoi':
        return _assign_voronoi(mask, seeds)
    if method != 'watershed':
        raise ValueError(f"method must be 'watershed' or 'voronoi', got {method}")

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[seeds[:, 0], seeds[:, 1]] = np.arange(1, seeds.shape[0] + 1)
    if prob is not None:
        relief = -prob
    else:
        relief = -ndi.distance_transform_edt(mask).astype(np.float32)
    try:
        from skimage.segmentation import watershed as _ws
        return _ws(relief, markers=markers, mask=mask).astype(np.int32)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning('watershed unavailable (%s); falling back to voronoi',
                       exc)
        return _assign_voronoi(mask, seeds)


def _assign_voronoi(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    pix = np.column_stack([rows, cols]).astype(np.float64)
    _, nearest = cKDTree(seeds.astype(np.float64)).query(pix, k=1)
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[rows, cols] = (nearest + 1).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# Step 4 - delineation + metrics
# ---------------------------------------------------------------------------
@dataclass
class TreeInstance:
    """One delineated tree in pixel coordinates (x=col, y=row in contour)."""
    label: int
    centroid_rc: Tuple[float, float]
    area_px: float
    contour: np.ndarray
    confidence: float = 1.0
    circularity: float = 0.0


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    return float(4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0


def instances_to_geometries(label_img: np.ndarray,
                            prob_map: Optional[np.ndarray] = None,
                            min_area_px: int = 9,
                            min_circularity: float = 0.0) -> List[TreeInstance]:
    """Extract one external contour + metrics per instance label.

    Args:
        label_img: int label image from :func:`assign_instances`.
        prob_map: optional probability surface for per-tree mean confidence.
        min_area_px: drop instances smaller than this (px^2).
        min_circularity: drop instances less circular than this in [0,1]
            (0 disables; useful to reject elongated non-crown blobs).
    """
    out: List[TreeInstance] = []
    labels = np.unique(label_img)
    labels = labels[labels != 0]
    for lab in labels:
        inst = label_img == lab
        if int(inst.sum()) < min_area_px:
            continue
        cnts, _ = cv2.findContours(inst.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        circ = _circularity(cnt)
        if circ < min_circularity:
            continue
        ys, xs = np.where(inst)
        conf = (float(prob_map[inst].mean())
                if prob_map is not None and inst.any() else 1.0)
        out.append(TreeInstance(int(lab), (float(ys.mean()), float(xs.mean())),
                                float(area), cnt, conf, circ))
    return out


# ---------------------------------------------------------------------------
# End-to-end (single array pair)
# ---------------------------------------------------------------------------
@dataclass
class PalmExtractionResult:
    label_img: np.ndarray
    trees: List[TreeInstance] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.trees)


def extract_trees(palm_mask: np.ndarray,
                  prob_map: Optional[np.ndarray] = None,
                  method: str = 'watershed',
                  blur_ksize: int = 5,
                  prob_threshold: float = 0.5,
                  min_area_px: int = 9,
                  min_circularity: float = 0.0,
                  min_seed_distance: int = 0) -> PalmExtractionResult:
    """Full mask(+optional heatmap) -> individual trees on one array pair.

    Works with or without a probability map (see :func:`find_seeds`).
    """
    seeds = find_seeds(palm_mask, prob_map, blur_ksize=blur_ksize,
                       prob_threshold=prob_threshold,
                       min_distance=min_seed_distance)
    labels = assign_instances(palm_mask, seeds, method=method,
                              prob_map=prob_map)
    trees = instances_to_geometries(labels, prob_map=prob_map,
                                    min_area_px=min_area_px,
                                    min_circularity=min_circularity)
    logger.info('extract_trees: %d seeds -> %d trees (method=%s, heatmap=%s)',
                seeds.shape[0], len(trees), method, prob_map is not None)
    return PalmExtractionResult(label_img=labels, trees=trees)


# Backwards-compatible alias for the earlier name.
extract_palms = extract_trees
