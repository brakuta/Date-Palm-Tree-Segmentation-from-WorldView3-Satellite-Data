# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for production hardening: fail-fast paths, validation, exit codes."""
import numpy as np
import pytest


# --- pipeline fail-fast ------------------------------------------------------
def test_segment_missing_input_raises_clearly(tmp_path):
    from palmseg.pipeline import segment
    with pytest.raises(FileNotFoundError, match='Input raster not found'):
        segment(model=None, raster_path=str(tmp_path / 'nope.tif'),
                out_dir=str(tmp_path))


def test_extract_missing_label_raises_clearly(tmp_path):
    from palmseg.pipeline import extract
    with pytest.raises(FileNotFoundError, match='Label/mask raster not found'):
        extract(str(tmp_path / 'nope.tif'), str(tmp_path / 'out.gpkg'))


def test_extract_missing_heatmap_raises_clearly(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from palmseg.pipeline import extract
    label = str(tmp_path / 'label.tif')
    arr = np.zeros((8, 8), dtype='uint8')
    with rasterio.open(label, 'w', driver='GTiff', height=8, width=8,
                       count=1, dtype='uint8') as dst:
        dst.write(arr, 1)
    with pytest.raises(FileNotFoundError, match='Heatmap raster not found'):
        extract(label, str(tmp_path / 'out.gpkg'),
                heatmap_path=str(tmp_path / 'nope_prob.tif'))


def test_extract_empty_mask_warns_and_writes_empty_layer(tmp_path, caplog):
    rasterio = pytest.importorskip('rasterio')
    pytest.importorskip('geopandas')
    import logging
    from palmseg.pipeline import extract
    label = str(tmp_path / 'label.tif')
    with rasterio.open(label, 'w', driver='GTiff', height=16, width=16,
                       count=1, dtype='uint8', crs='EPSG:32640') as dst:
        dst.write(np.zeros((16, 16), dtype='uint8'), 1)
    out_vec = str(tmp_path / 'trees.gpkg')
    with caplog.at_level(logging.WARNING, logger='palmseg.pipeline'):
        res = extract(label, out_vec)
    assert res.count == 0
    assert any('no pixels of class' in r.message for r in caplog.records)


# --- tiled inference config validation --------------------------------------
def test_tile_infer_config_validation(tmp_path):
    from palmseg.inference.tiled import infer_raster, TileInferConfig
    bad = [TileInferConfig(tile_size=0),
           TileInferConfig(overlap=-1),
           TileInferConfig(tile_size=128, overlap=128)]
    for cfg in bad:
        with pytest.raises(ValueError):
            infer_raster(model=None, raster_path=str(tmp_path / 'x.tif'),
                         out_label_path=str(tmp_path / 'y.tif'), cfg=cfg)


def test_tile_infer_missing_raster(tmp_path):
    from palmseg.inference.tiled import infer_raster
    with pytest.raises(FileNotFoundError, match='Input raster not found'):
        infer_raster(model=None, raster_path=str(tmp_path / 'nope.tif'),
                     out_label_path=str(tmp_path / 'y.tif'))


# --- CLI behaviour -----------------------------------------------------------
def test_cli_download_requires_id_or_flag():
    from palmseg.cli import main
    with pytest.raises(SystemExit):
        main(['download'])


def test_cli_extract_missing_label_exits_cleanly():
    from palmseg.cli import main
    with pytest.raises(SystemExit):
        main(['extract', '--label', 'definitely_missing.tif',
              '--out', 'x.gpkg'])


def test_cli_unknown_model_id_friendly():
    from palmseg.weights_manifest import resolve_config
    with pytest.raises(KeyError, match='Unknown model id'):
        resolve_config('not_a_model')


# --- loader verification ------------------------------------------------------
def test_verified_load_detects_mismatch():
    torch = pytest.importorskip('torch')
    from palmseg.loader import _verified_load

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(8, 4, 3)
            self.head = torch.nn.Conv2d(4, 2, 1)

    model = Tiny()
    good = {k: v.clone() for k, v in model.state_dict().items()}
    _verified_load(model, good, 'good.safetensors')        # must not raise

    missing = {k: v for k, v in good.items() if 'head' not in k}
    with pytest.raises(RuntimeError, match='missing from the checkpoint'):
        _verified_load(model, missing, 'missing.safetensors')

    wrong = dict(good)
    wrong['backbone.weight'] = torch.randn(4, 3, 3, 3)     # wrong in_channels
    with pytest.raises(RuntimeError, match='shape mismatch'):
        _verified_load(model, wrong, 'wrong.safetensors')


def test_resolve_device_auto_falls_back_to_cpu():
    torch = pytest.importorskip('torch')
    from palmseg.loader import resolve_device
    dev = resolve_device('auto')
    assert dev in ('cpu',) or dev.startswith('cuda')
    if not torch.cuda.is_available():
        assert dev == 'cpu'
        with pytest.raises(RuntimeError, match='CUDA is not available'):
            resolve_device('cuda:0')


# --- NaN-nodata handling in tiled reads --------------------------------------
def test_read_tile_nan_nodata(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from palmseg.inference.tiled import _read_tile
    path = str(tmp_path / 'nan_nodata.tif')
    arr = np.full((3, 16, 16), 0.5, dtype='float32')
    arr[:, :4, :] = np.nan                          # nodata strip
    with rasterio.open(path, 'w', driver='GTiff', height=16, width=16,
                       count=3, dtype='float32', nodata=float('nan')) as dst:
        dst.write(arr)
    with rasterio.open(path) as ds:
        tile, valid = _read_tile(ds, 0, 0, 16)
    assert not valid[:4, :].any() and valid[4:, :].all()
    assert np.isfinite(tile).all()                  # no NaN reaches the model
