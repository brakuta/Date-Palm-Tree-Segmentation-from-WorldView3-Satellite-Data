# Copyright (c) 2024. Licensed under Apache-2.0.
"""Manifest of released palm WV-3 models and a Hugging Face download helper.

Each entry maps a short model id -> (config file, checkpoint filename,
modality, family, heatmap capability). Weights are published as ``.safetensors``
(tensor-only, no pickle), which is the format the Hugging Face security scanner
reports as safe and which ``palmseg.loader`` loads directly.

To (re)publish weights: convert the trained ``.pth`` checkpoints with
``python scripts/to_safetensors.py weights/`` and upload the resulting
``.safetensors`` files under the names below. Set ``HF_REPO_ID`` to the model
repo once created.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

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
    # === multispectral (8-band) ============================================
    'segformer_b3_ms':        ModelEntry('configs/segformer_b3_ms.py',
                                         'segformer_b3_ms.safetensors', 'ms',
                                         'segformer', True),
    'segformer_b5_ms':        ModelEntry('configs/segformer_b5_ms.py',
                                         'segformer_b5_ms.safetensors', 'ms',
                                         'segformer', True),
    'upernet_swin_s_ms':      ModelEntry('configs/upernet_swin_s_ms.py',
                                         'upernet_swin_s_ms.safetensors', 'ms',
                                         'upernet_swin', True),
    'upernet_swin_b_ms':      ModelEntry('configs/upernet_swin_b_ms.py',
                                         'upernet_swin_b_ms.safetensors', 'ms',
                                         'upernet_swin', True),
    'upernet_vit_deit_s_ms':  ModelEntry('configs/upernet_vit_deit_s_ms.py',
                                         'upernet_vit_deit_s_ms.safetensors',
                                         'ms', 'upernet_vit', True),
    'uniformer_fpn_global_ms': ModelEntry('configs/uniformer_base_ms.py',
                                          'uniformer_fpn_global_ms.safetensors',
                                          'ms', 'uniformer', True),
    'uniformer_xs_ms':        ModelEntry('configs/uniformer_xs_ms.py',
                                         'uniformer_xs_ms.safetensors', 'ms',
                                         'uniformer', True),
    'mask2former_swin_b_ms':  ModelEntry('configs/mask2former_swin_b_ms.py',
                                         'mask2former_swin_b_ms.safetensors',
                                         'ms', 'mask2former', False),
    'mask2former_swin_s_ms':  ModelEntry('configs/mask2former_swin_s_ms.py',
                                         'mask2former_swin_s_ms.safetensors',
                                         'ms', 'mask2former', False),
    # === RGB (3-band) ======================================================
    'segformer_b3_rgb':       ModelEntry('configs/segformer_b3_rgb.py',
                                         'segformer_b3_rgb.safetensors', 'rgb',
                                         'segformer', True),
    'upernet_swin_t_rgb':     ModelEntry('configs/upernet_swin_t_rgb.py',
                                         'upernet_swin_t_rgb.safetensors', 'rgb',
                                         'upernet_swin', True),
    'upernet_vit_deit_s_rgb': ModelEntry('configs/upernet_vit_deit_s_rgb.py',
                                         'upernet_vit_deit_s_rgb.safetensors',
                                         'rgb', 'upernet_vit', True),
    'uniformer_xs_rgb':       ModelEntry('configs/uniformer_xs_rgb.py',
                                         'uniformer_xs_rgb.safetensors', 'rgb',
                                         'uniformer', True),
    'mask2former_swin_t_rgb': ModelEntry('configs/mask2former_swin_t_rgb.py',
                                         'mask2former_swin_t_rgb.safetensors',
                                         'rgb', 'mask2former', False),
}


def download_checkpoint(model_id: str, cache_dir: str = 'weights',
                        repo_id: Optional[str] = None) -> str:
    """Download a released checkpoint from Hugging Face Hub.

    Returns the local path to the downloaded checkpoint file
    (``.safetensors`` for the released models). Requires ``huggingface_hub``.
    """
    if model_id not in MODELS:
        raise KeyError(f'Unknown model id "{model_id}". '
                       f'Known: {list(MODELS)}')
    entry = MODELS[model_id]
    repo_id = repo_id or HF_REPO_ID
    if not repo_id or repo_id.startswith('CHANGE_ME'):
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
    if model_id not in MODELS:
        raise KeyError(f'Unknown model id "{model_id}". Known: {list(MODELS)}')
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
