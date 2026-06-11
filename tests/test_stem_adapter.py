# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Tests for the input-stem adapter (no GPU needed)."""
import torch
import pytest

from palmseg.tools import adapt_input_stem as A


@pytest.mark.parametrize('key', [
    'backbone.layers.0.0.projection.weight',      # SegFormer
    'backbone.patch_embed.projection.weight',     # Swin / ViT
])
def test_stem_autodetect_and_bandmap(key, tmp_path):
    ckpt = {'state_dict': {
        key: torch.randn(64, 3, 7, 7),
        'decode_head.conv_seg.weight': torch.randn(2, 64, 1, 1),
    }}
    assert A.find_stem_key(ckpt['state_dict']) == key
    src = tmp_path / 'in.pth'
    dst = tmp_path / 'out.pth'
    torch.save(ckpt, src)
    A.adapt_checkpoint(str(src), str(dst), new_in=8, strategy='band_map')
    w = torch.load(dst, weights_only=False)['state_dict'][key]
    assert tuple(w.shape) == (64, 8, 7, 7)
    orig = ckpt['state_dict'][key]
    # RGB kernels placed at WV-3 band indices: Red=4<-R0, Blue=1<-B2, Green=2<-G1
    assert torch.allclose(w[:, 4:5], orig[:, 0:1])
    assert torch.allclose(w[:, 1:2], orig[:, 2:3])
    assert torch.allclose(w[:, 2:3], orig[:, 1:2])
    # non-RGB band filled with channel mean
    assert torch.allclose(w[:, 0:1], orig.mean(dim=1, keepdim=True))


def test_mean_strategy_scale(tmp_path):
    key = 'backbone.patch_embed.projection.weight'
    ckpt = {'state_dict': {key: torch.randn(32, 3, 4, 4)}}
    src = tmp_path / 'i.pth'; dst = tmp_path / 'o.pth'
    torch.save(ckpt, src)
    A.adapt_checkpoint(str(src), str(dst), new_in=8, strategy='mean', scale=True)
    w = torch.load(dst, weights_only=False)['state_dict'][key]
    assert tuple(w.shape) == (32, 8, 4, 4)


def test_zero_strategy_keeps_rgb(tmp_path):
    key = 'backbone.patch_embed.projection.weight'
    ckpt = {'state_dict': {key: torch.randn(16, 3, 4, 4)}}
    src = tmp_path / 'i.pth'; dst = tmp_path / 'o.pth'
    torch.save(ckpt, src)
    A.adapt_checkpoint(str(src), str(dst), new_in=8, strategy='zero')
    w = torch.load(dst, weights_only=False)['state_dict'][key]
    assert torch.allclose(w[:, :3], ckpt['state_dict'][key])
    assert torch.count_nonzero(w[:, 3:]) == 0
