# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Validate that every shipped config parses and has the correct modality
contract. Catches _base_ path errors and missing data_preprocessor."""
import glob
import pytest

mmengine = pytest.importorskip('mmengine')
from mmengine.config import Config

CONFIGS = sorted(glob.glob('configs/*.py'))
CONFIGS = [c for c in CONFIGS if '_base_' not in c]


@pytest.mark.parametrize('cfg_path', CONFIGS)
def test_config_parses_and_modality(cfg_path):
    cfg = Config.fromfile(cfg_path, import_custom_modules=False)
    assert 'model' in cfg
    dp = cfg.model.get('data_preprocessor')
    assert dp is not None, 'data_preprocessor missing from model (base merge?)'
    mean = dp['mean']
    is_ms = cfg_path.endswith('_ms.py')
    bb = cfg.model['backbone']
    inch = bb.get('in_channels', bb.get('in_chans'))
    if is_ms:
        assert all(v == 0 for v in mean), f'MS mean must be zeros, got {mean}'
        assert dp['bgr_to_rgb'] is False
        assert inch == 8
    else:
        assert mean[0] > 100, f'RGB mean must be ImageNet-scale, got {mean}'
        assert dp['bgr_to_rgb'] is True
        assert inch == 3
    # base merge brought in the training machinery
    for key in ('train_dataloader', 'val_dataloader', 'optim_wrapper',
                'param_scheduler', 'visualizer'):
        assert key in cfg, f'{key} missing (base merge failed)'
