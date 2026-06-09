# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for instance extraction and vectorisation (no GPU / no MMSeg needed)."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from palmseg.postprocess import individual_trees as it
from palmseg.postprocess import vectorize as vz
from palmseg import pipeline


H = W = 200
PX = 0.31
TRANSFORM = from_origin(400000.0, 2700000.0, PX, PX)
CRS = 'EPSG:32640'


def _disk(cy, cx, r, peak=1.0):
    yy, xx = np.ogrid[:H, :W]
    return np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1) * peak


@pytest.fixture
def scene():
    prob = np.maximum.reduce([
        _disk(60, 60, 22), _disk(60, 95, 22), _disk(140, 140, 26)
    ]).astype(np.float32)
    mask = prob > 0.15
    return mask, prob


def test_seeds_with_heatmap(scene):
    mask, prob = scene
    seeds = it.find_seeds(mask, prob, prob_threshold=0.2)
    assert seeds.shape[0] == 3


def test_seeds_without_heatmap(scene):
    mask, _ = scene
    seeds = it.find_seeds(mask, None)
    # distance-transform seeding still finds >=3 crown centres
    assert seeds.shape[0] >= 3


@pytest.mark.parametrize('method', ['watershed', 'voronoi'])
def test_touching_crowns_split(scene, method):
    mask, prob = scene
    res = it.extract_trees(mask, prob, method=method, prob_threshold=0.2,
                           min_area_px=20)
    assert res.count >= 3


def test_mask_only_extraction(scene):
    mask, _ = scene
    res = it.extract_trees(mask, None, method='watershed', min_area_px=20)
    assert res.count >= 3


def test_circularity_filter_keeps_round(scene):
    mask, prob = scene
    res = it.extract_trees(mask, prob, prob_threshold=0.2, min_area_px=20,
                           min_circularity=0.6)
    assert res.count >= 3
    assert all(t.circularity >= 0.6 for t in res.trees)


def test_seed_nms(scene):
    mask, prob = scene
    s_all = it.find_seeds(mask, prob, prob_threshold=0.2)
    s_nms = it.find_seeds(mask, prob, prob_threshold=0.2, min_distance=10)
    assert s_nms.shape[0] <= s_all.shape[0]


def test_validation_rejects_3d():
    with pytest.raises(ValueError):
        it.find_seeds(np.zeros((4, 4, 4)))


def test_prob_shape_mismatch_raises(scene):
    mask, _ = scene
    with pytest.raises(ValueError):
        it.assign_instances(mask, np.array([[10, 10]]),
                            prob_map=np.zeros((10, 10)))


def test_georeferencing_in_extent(scene, tmp_path):
    mask, prob = scene
    res = it.extract_trees(mask, prob, prob_threshold=0.2, min_area_px=20)
    gdf = vz.instances_to_gdf(res.trees, TRANSFORM, CRS, geometry='circle')
    assert len(gdf) == res.count
    c = gdf.geometry.iloc[0].centroid
    assert 400000 <= c.x <= 400000 + W * PX
    assert 2700000 - H * PX <= c.y <= 2700000
    # crown diameters at 0.31 m GSD should be physically plausible (5-20 m)
    assert gdf['diam_m'].between(5, 20).all()


def test_diameter_matches_area(scene):
    mask, prob = scene
    res = it.extract_trees(mask, prob, prob_threshold=0.2, min_area_px=20)
    gdf = vz.instances_to_gdf(res.trees, TRANSFORM, CRS, geometry='circle')
    # d = 2*sqrt(area_m2/pi); area_m2 = area_px * px^2
    for _, row in gdf.iterrows():
        area_m2 = row['area_px'] * PX ** 2
        expected = 2 * np.sqrt(area_m2 / np.pi)
        assert abs(row['diam_m'] - expected) < 0.1


def test_instance_vs_naive_mask(scene, tmp_path):
    """Instance extraction separates touching crowns; naive mask vectorisation
    merges them. This is the core value of the seed step."""
    mask, prob = scene
    lab = tmp_path / 'L.tif'
    with rasterio.open(lab, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs=CRS, transform=TRANSFORM) as d:
        d.write(mask.astype('uint8'), 1)
    naive = vz.mask_to_gdf(str(lab), class_value=1)
    res = it.extract_trees(mask, prob, prob_threshold=0.2, min_area_px=20)
    assert res.count > len(naive)  # 3 instances vs 2 merged blobs


def test_pipeline_extract_modes(scene, tmp_path):
    mask, prob = scene
    lab = tmp_path / 'L.tif'
    hm = tmp_path / 'P.tif'
    with rasterio.open(lab, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs=CRS, transform=TRANSFORM) as d:
        d.write(mask.astype('uint8'), 1)
    with rasterio.open(hm, 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='float32', crs=CRS, transform=TRANSFORM) as d:
        d.write(prob, 1)
    e_hm = pipeline.extract(str(lab), str(tmp_path / 'with.gpkg'),
                            heatmap_path=str(hm), prob_threshold=0.2,
                            min_area_px=20)
    e_mask = pipeline.extract(str(lab), str(tmp_path / 'mask.gpkg'),
                              heatmap_path=None, min_area_px=20)
    assert e_hm.count >= 3 and e_mask.count >= 3
    assert e_hm.heatmap_used is not None
    assert e_mask.heatmap_used is None


def test_deduplicate_overlaps():
    import geopandas as gpd
    from shapely.geometry import Point
    from palmseg.postprocess.vectorize import deduplicate_overlaps
    g = gpd.GeoDataFrame(
        {'score': [0.9, 0.5, 0.8],
         'geometry': [Point(0, 0).buffer(5), Point(0.5, 0).buffer(5),
                      Point(100, 100).buffer(5)]},
        crs='EPSG:32640')
    out = deduplicate_overlaps(g, iou_threshold=0.5)
    assert len(out) == 2
    assert 0.5 not in out['score'].tolist()  # weaker overlapping crown removed


def test_single_blob_is_one_tree():
    """Regression: a solid blob must not over-segment from distance-transform
    plateau maxima."""
    m = np.zeros((50, 50), bool)
    m[10:30, 10:30] = True
    res = it.extract_trees(m, None, min_area_px=5)
    assert res.count == 1


def test_two_separate_blobs_mask_only():
    m = np.zeros((100, 100), bool)
    m[10:30, 10:30] = True
    m[60:85, 60:85] = True
    res = it.extract_trees(m, None, min_area_px=5)
    assert res.count == 2


def test_border_crown_valid_geometry():
    m = np.zeros((50, 50), bool)
    m[0:15, 0:15] = True
    res = it.extract_trees(m, None, min_area_px=5)
    gdf = vz.instances_to_gdf(res.trees, TRANSFORM, CRS, geometry='polygon')
    assert res.count == 1
    assert gdf.geometry.is_valid.all()


def test_empty_mask_no_crash():
    res = it.extract_trees(np.zeros((40, 40), bool), None, min_area_px=5)
    gdf = vz.instances_to_gdf(res.trees, TRANSFORM, CRS)
    assert res.count == 0 and len(gdf) == 0
