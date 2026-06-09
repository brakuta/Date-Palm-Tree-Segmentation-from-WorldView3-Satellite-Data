# Copyright (c) 2024. Licensed under Apache-2.0.
"""Copy trained checkpoints to the filenames expected by the weights manifest.

Reads palmseg.weights_manifest.MODELS and, for each model id, copies a source
checkpoint to ``<out_dir>/<manifest filename>``. Source paths are given on the
command line as ``model_id=path`` pairs, or discovered from a work_dirs tree by
matching the config name.

Examples:
    python scripts/rename_weights.py --out weights \\
        segformer_b5_ms=work_dirs/ALL-segformer.../iter_100000.pth \\
        segformer_b5_rgb=work_dirs/segformer.../iter_100000.pth
"""
from __future__ import annotations

import argparse
import os
import shutil

from palmseg.weights_manifest import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='*', help='model_id=path/to/checkpoint.pth')
    ap.add_argument('--out', default='weights')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not args.pairs:
        print('Known model ids and their target filenames:')
        for mid, e in MODELS.items():
            print(f'  {mid:24s} -> {e.checkpoint}')
        print('\nUsage: python scripts/rename_weights.py --out weights '
              'model_id=path.pth ...')
        return

    for pair in args.pairs:
        if '=' not in pair:
            raise SystemExit(f'expected model_id=path, got: {pair}')
        mid, src = pair.split('=', 1)
        if mid not in MODELS:
            raise SystemExit(f'unknown model id {mid}; known: {list(MODELS)}')
        if not os.path.isfile(src):
            raise SystemExit(f'source not found: {src}')
        dst = os.path.join(args.out, MODELS[mid].checkpoint)
        shutil.copy2(src, dst)
        print(f'{mid}: {src} -> {dst}')


if __name__ == '__main__':
    main()
