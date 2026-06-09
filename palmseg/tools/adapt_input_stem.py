# Copyright (c) 2024. Licensed under Apache-2.0.
"""Adapt a backbone input stem from 3 channels to N channels.

Creates an N-channel pretrained checkpoint to initialise multispectral
training from a 3-channel (ImageNet) backbone. Inference on the released models
does not need this step; their checkpoints already contain an N-channel stem.

The stem key is detected automatically across backbone families (SegFormer
``layers.0.0.projection``; Swin/ViT ``patch_embed.projection``). Strategies:

  band_map  place the pretrained R/G/B kernels at their WorldView-3 band
            indices and fill the remaining bands with the channel mean.
  mean      tile the channel mean across all N channels (with optional
            magnitude rescaling).
  zero      keep the 3 pretrained kernels and zero the extra channels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Optional, Sequence

import torch


# WV-3 8-band order: 0 Coastal,1 Blue,2 Green,3 Yellow,4 Red,5 RedEdge,6 NIR1,7 NIR2
# Pretrained stem channel order is RGB: source 0=R, 1=G, 2=B.
WV3_8BAND_RGB_MAP: Sequence[Optional[int]] = (
    None,  # 0 Coastal
    2,     # 1 Blue  <- pretrained B
    1,     # 2 Green <- pretrained G
    None,  # 3 Yellow
    0,     # 4 Red   <- pretrained R
    None,  # 5 Red Edge
    None,  # 6 NIR-1
    None,  # 7 NIR-2
)


def _unwrap(ckpt):
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], ckpt, 'state_dict'
    return ckpt, ckpt, None


def find_stem_key(state_dict, orig_in: int = 3) -> str:
    """Locate the backbone input-stem conv weight (4-D, in_dim == orig_in)."""
    cands = [
        k for k, v in state_dict.items()
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == orig_in
        and 'backbone' in k
        and (k.endswith('projection.weight') or 'patch_embed' in k
             or k.endswith('proj.weight') or k.endswith('stem.0.weight'))
    ]
    if not cands:
        raise KeyError(
            f'No backbone stem conv with in_dim={orig_in} found. '
            f'Pass --stem-key explicitly.')
    cands.sort(key=lambda k: (0 if 'patch_embed' in k else 1, k))
    return cands[0]


def adapt_stem_weight(w: torch.Tensor, new_in: int, strategy: str = 'band_map',
                      band_map: Optional[Sequence[Optional[int]]] = WV3_8BAND_RGB_MAP,
                      scale: bool = False) -> torch.Tensor:
    E, orig_in, k1, k2 = w.shape
    mean_kernel = w.mean(dim=1, keepdim=True)
    new_w = torch.zeros(E, new_in, k1, k2, dtype=w.dtype, device=w.device)

    if strategy == 'mean':
        new_w = mean_kernel.repeat(1, new_in, 1, 1)
        if scale:
            new_w = new_w * (orig_in / new_in)
    elif strategy == 'zero':
        n = min(orig_in, new_in)
        new_w[:, :n] = w[:, :n]
    elif strategy == 'band_map':
        if band_map is None or len(band_map) != new_in:
            raise ValueError(
                f'band_map of length {new_in} required for strategy="band_map"')
        for out_idx, src in enumerate(band_map):
            if src is None:
                new_w[:, out_idx:out_idx + 1] = mean_kernel
            else:
                if not (0 <= src < orig_in):
                    raise ValueError(f'band_map[{out_idx}]={src} out of range')
                new_w[:, out_idx:out_idx + 1] = w[:, src:src + 1]
        if scale:
            new_w = new_w * (orig_in / new_in)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    return new_w


def adapt_checkpoint(in_path: str, out_path: str, new_in: int = 8,
                     strategy: str = 'band_map', stem_key: Optional[str] = None,
                     band_map: Optional[Sequence[Optional[int]]] = WV3_8BAND_RGB_MAP,
                     scale: bool = False) -> str:
    ckpt = torch.load(in_path, map_location='cpu')
    state_dict, container, sd_key = _unwrap(deepcopy(ckpt))
    key = stem_key or find_stem_key(state_dict, orig_in=3)
    old_w = state_dict[key]
    new_w = adapt_stem_weight(old_w, new_in, strategy, band_map, scale)
    state_dict[key] = new_w
    to_save = container if sd_key is not None else state_dict
    if sd_key is not None:
        container[sd_key] = state_dict
    torch.save(to_save, out_path)
    print(f'[adapt_input_stem] stem key : {key}')
    print(f'[adapt_input_stem] {tuple(old_w.shape)} -> {tuple(new_w.shape)} '
          f'(strategy={strategy}, scale={scale})')
    print(f'[adapt_input_stem] saved    : {out_path}')
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description='Adapt a 3-channel backbone stem to N channels.')
    p.add_argument('in_path')
    p.add_argument('out_path')
    p.add_argument('--new-in', type=int, default=8)
    p.add_argument('--strategy', choices=['band_map', 'mean', 'zero'],
                   default='band_map')
    p.add_argument('--stem-key', default=None)
    p.add_argument('--scale', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    bmap = WV3_8BAND_RGB_MAP if (a.strategy == 'band_map' and a.new_in == 8) else None
    adapt_checkpoint(a.in_path, a.out_path, a.new_in, a.strategy,
                     a.stem_key, bmap, a.scale)
