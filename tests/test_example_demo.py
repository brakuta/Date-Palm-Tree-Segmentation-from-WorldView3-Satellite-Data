# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Guards the quickstart: demo data generation + extraction must work."""
import importlib.util
import os


from palmseg.pipeline import extract


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        'make_demo_data', 'examples/make_demo_data.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_demo_generation_and_extraction(tmp_path):
    demo = _load_demo_module()
    out = str(tmp_path / 'demo')
    demo.main(out_dir=out, size=256, n_trees=20, seed=1)
    for f in ('scene_rgb.tif', 'scene_label.tif', 'scene_prob.tif'):
        assert os.path.exists(os.path.join(out, f))
    res = extract(os.path.join(out, 'scene_label.tif'),
                  str(tmp_path / 'trees.gpkg'),
                  heatmap_path=os.path.join(out, 'scene_prob.tif'),
                  min_area_px=20, min_seed_distance=6)
    assert res.count > 0
