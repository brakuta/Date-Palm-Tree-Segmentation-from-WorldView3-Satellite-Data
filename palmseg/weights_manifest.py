# Copyright (c) 2024. Licensed under Apache-2.0.
"""Manifest of released palm WV-3 models and a Hugging Face download helper.

Each entry maps a short model id -> (config file, checkpoint filename,
modality). The Hugging Face repo id is a placeholder to be set once the
weights are uploaded; the download helper resolves files from it.

Fill HF_REPO_ID after creating the model repo, e.g.:
    huggingface-cli repo create palm-wv3-models --type model
and upload the .pth files with the names in CHECKPOINTS below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

# <-- set this to your uploaded HF model repo, e.g. 'mbgibril/palm-wv3-models'
HF_REPO_ID = 'brakuta/date-palm-wv3-models'


@dataclass(frozen=True)
class ModelEntry:
    config: str            # path under configs/
    checkpoint: str        # filename within the HF repo
    modality: str          # 'ms' | 'rgb'
    family: str
    heatmap_capable: bool  # conv-seg heads support the tree-counting heatmap


MODELS: Dict[str, ModelEntry] = {
    # --- multispectral (8-band) ---
    'segformer_b5_ms':      ModelEntry('configs/segformer_b5_ms.py',
                                       'segformer_b5_ms.pth', 'ms',
                                       'segformer', True),
    'upernet_swin_b_ms':    ModelEntry('configs/upernet_swin_b_ms.py',
                                       'upernet_swin_b_ms.pth', 'ms',
                                       'upernet_swin', True),
    'upernet_vit_deit_s_ms': ModelEntry('configs/upernet_vit_deit_s_ms.py',
                                        'upernet_vit_deit_s_ms.pth', 'ms',
                                        'upernet_vit', True),
    'uniformer_base_ms':    ModelEntry('configs/uniformer_base_ms.py',
                                       'uniformer_base_ms.pth', 'ms',
                                       'uniformer', True),
    'mask2former_swin_s_ms': ModelEntry('configs/mask2former_swin_s_ms.py',
                                        'mask2former_swin_s_ms.pth', 'ms',
                                        'mask2former', False),
    # --- RGB ---
    'segformer_b5_rgb':     ModelEntry('configs/segformer_b5_rgb.py',
                                       'segformer_b5_rgb.pth', 'rgb',
                                       'segformer', True),
    'upernet_swin_b_rgb':   ModelEntry('configs/upernet_swin_b_rgb.py',
                                       'upernet_swin_b_rgb.pth', 'rgb',
                                       'upernet_swin', True),
    'upernet_vit_deit_s_rgb': ModelEntry('configs/upernet_vit_deit_s_rgb.py',
                                         'upernet_vit_deit_s_rgb.pth', 'rgb',
                                         'upernet_vit', True),
    'uniformer_base_rgb':   ModelEntry('configs/uniformer_base_rgb.py',
                                       'uniformer_base_rgb.pth', 'rgb',
                                       'uniformer', True),
    'mask2former_swin_s_rgb': ModelEntry('configs/mask2former_swin_s_rgb.py',
                                         'mask2former_swin_s_rgb.pth', 'rgb',
                                         'mask2former', False),
    # --- light / standard variants (release if trained and validated) ---
    'segformer_b0_ms':      ModelEntry('configs/segformer_b0_ms.py',
                                       'segformer_b0_ms.pth', 'ms',
                                       'segformer', True),
    'segformer_b0_rgb':     ModelEntry('configs/segformer_b0_rgb.py',
                                       'segformer_b0_rgb.pth', 'rgb',
                                       'segformer', True),
    'segformer_b2_ms':      ModelEntry('configs/segformer_b2_ms.py',
                                       'segformer_b2_ms.pth', 'ms',
                                       'segformer', True),
    'segformer_b2_rgb':     ModelEntry('configs/segformer_b2_rgb.py',
                                       'segformer_b2_rgb.pth', 'rgb',
                                       'segformer', True),
    'upernet_swin_t_ms':    ModelEntry('configs/upernet_swin_t_ms.py',
                                       'upernet_swin_t_ms.pth', 'ms',
                                       'upernet_swin', True),
    'upernet_swin_t_rgb':   ModelEntry('configs/upernet_swin_t_rgb.py',
                                       'upernet_swin_t_rgb.pth', 'rgb',
                                       'upernet_swin', True),
}


def download_checkpoint(model_id: str, cache_dir: str = 'weights',
                        repo_id: str = None) -> str:
    """Download a released checkpoint from Hugging Face Hub.

    Returns the local path to the .pth. Requires ``huggingface_hub``.
    """
    if model_id not in MODELS:
        raise KeyError(f'Unknown model id "{model_id}". '
                       f'Known: {list(MODELS)}')
    entry = MODELS[model_id]
    repo_id = repo_id or HF_REPO_ID
    if repo_id.startswith('CHANGE_ME'):
        raise RuntimeError(
            'HF_REPO_ID is not set. Edit palmseg/weights_manifest.py or pass '
            'repo_id=... once the weights are uploaded to Hugging Face.')
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError('pip install huggingface_hub') from exc
    os.makedirs(cache_dir, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=entry.checkpoint,
                           local_dir=cache_dir)
    return path


def resolve_config(model_id: str) -> str:
    return MODELS[model_id].config


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Download released palm weights.')
    ap.add_argument('model_id', nargs='?', help='e.g. segformer_b5_ms')
    ap.add_argument('--all', action='store_true', help='download every model')
    ap.add_argument('--cache-dir', default='weights')
    ap.add_argument('--repo-id', default=None)
    ap.add_argument('--list', action='store_true', help='list model ids')
    args = ap.parse_args()
    if args.list or (not args.model_id and not args.all):
        for k, v in MODELS.items():
            tag = 'heatmap' if v.heatmap_capable else 'mask-only'
            print(f'  {k:24s} {v.modality:3s} {v.family:14s} [{tag}]')
        raise SystemExit(0)
    ids = list(MODELS) if args.all else [args.model_id]
    for mid in ids:
        p = download_checkpoint(mid, args.cache_dir, args.repo_id)
        print(f'{mid} -> {p}')
