#!/usr/bin/env python3
# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Convert PyTorch ``.pth`` checkpoints to ``.safetensors``.

Why: a ``.pth`` file is a Python pickle. Loading one can execute arbitrary code,
so the Hugging Face security scanner flags pickled weights as "Unsafe" or
"Suspicious". The ``safetensors`` format stores only raw tensors plus a small
JSON header, with no executable payload, so the scanner reports the file as
safe and loading cannot run code.

What this does, per input checkpoint:
  1. extracts the parameter tensors (the ``state_dict``), discarding the
     pickled training ``meta`` (optimizer state, config object, argparse
     namespace) that is both unnecessary for inference and the usual cause of
     the scanner warning;
  2. strips a leading ``module.`` from keys (left by DataParallel), if present;
  3. moves every tensor to CPU and clones it to contiguous storage so that no
     two keys share memory (``safetensors`` rejects shared storage);
  4. writes ``<name>.safetensors`` next to (or under ``--out``) the source.

The state dict produced here is exactly what ``palmseg.loader`` loads, so the
released models become drop-in safetensors with no change to inference.

Examples
--------
  # Convert every .pth in weights/ in place (writes .safetensors alongside):
  python scripts/to_safetensors.py weights/

  # Convert into a separate folder and remove the source .pth afterwards:
  python scripts/to_safetensors.py weights/ --out weights_st --delete-source
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
from typing import Dict

import torch


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    """Return the parameter tensor dict from a raw mmengine/torch checkpoint."""
    sd = ckpt['state_dict'] if (isinstance(ckpt, dict)
                                and 'state_dict' in ckpt) else ckpt
    if not isinstance(sd, dict):
        raise TypeError('checkpoint does not contain a state_dict mapping')
    clean: Dict[str, torch.Tensor] = {}
    skipped = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            skipped += 1                       # drop non-tensor entries
            continue
        key = k[len('module.'):] if k.startswith('module.') else k
        # CPU + clone + contiguous breaks any shared storage and detaches grad.
        clean[key] = v.detach().to('cpu').clone().contiguous()
    if skipped:
        print(f'    note: dropped {skipped} non-tensor entry(ies)')
    return clean


def convert_one(src: str, out_dir: str | None, delete_source: bool) -> str:
    from safetensors.torch import save_file
    # weights_only=False is required to read mmengine checkpoints, whose `meta`
    # holds pickled config objects. This is the very pickle risk being removed:
    # it runs only on your own local file, and the output is pickle-free.
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    state = _extract_state_dict(ckpt)
    stem = osp.splitext(osp.basename(src))[0]
    dst_dir = out_dir or osp.dirname(osp.abspath(src))
    os.makedirs(dst_dir, exist_ok=True)
    dst = osp.join(dst_dir, f'{stem}.safetensors')
    # A tiny metadata block is allowed (strings only) and is not executable.
    save_file(state, dst, metadata={'format': 'pt', 'source': 'palmseg'})
    n_par = sum(t.numel() for t in state.values())
    print(f'  {osp.basename(src)} -> {osp.basename(dst)} '
          f'({len(state)} tensors, {n_par/1e6:.1f}M params)')
    if delete_source:
        os.remove(src)
        print(f'    removed source: {src}')
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Convert .pth checkpoints to .safetensors (clears the '
                    'Hugging Face unsafe-pickle warning).')
    ap.add_argument('inputs', nargs='+',
                    help='one or more .pth files, or a folder of .pth files')
    ap.add_argument('--out', default=None,
                    help='output folder (default: alongside each source)')
    ap.add_argument('--delete-source', action='store_true',
                    help='remove each .pth after a successful conversion')
    args = ap.parse_args()

    try:
        import safetensors  # noqa: F401
    except ImportError:
        raise SystemExit('pip install safetensors')

    files = []
    for item in args.inputs:
        if osp.isdir(item):
            files.extend(sorted(glob.glob(osp.join(item, '*.pth'))))
        elif item.endswith('.pth'):
            files.append(item)
        else:
            print(f'skipping non-.pth input: {item}')
    if not files:
        raise SystemExit('no .pth files found to convert')

    print(f'converting {len(files)} checkpoint(s):')
    for src in files:
        convert_one(src, args.out, args.delete_source)
    print('done.')


if __name__ == '__main__':
    main()
