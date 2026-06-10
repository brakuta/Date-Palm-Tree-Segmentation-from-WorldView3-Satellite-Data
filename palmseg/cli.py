#!/usr/bin/env python3
# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""palmseg - date palm segmentation & individual-tree mapping.

Three independent operating modes (each runs on its own):

  segment              (a) model -> class-label raster (+ optional heatmap)
  extract              (b) existing raster(s) -> individual-tree polygons
                           (heatmap OPTIONAL; runs on a binary mask alone)
  segment-and-extract  (c) (a) then (b) end-to-end

Plus utilities:
  inspect      show a checkpoint's channels / classes / modality
  selftest     run a model on a synthetic tile and sanity-check output
  adapt-stem   expand a 3-channel pretrained stem to N channels (fine-tune prep)
  prep-check   validate a prepared dataset before fine-tuning
  download     fetch released weights from Hugging Face

Examples
--------
  # (a) segmentation only
  palmseg segment --model segformer_b5_ms --checkpoint w/seg_b5_ms.pth \
                  --input scene.tif --out out/            # writes label + heatmap
  palmseg segment ... --no-heatmap                        # label only

  # (b) extraction only, from rasters you already have
  palmseg extract --label out/scene_label.tif --heatmap out/scene_prob.tif \
                  --out out/scene_trees.gpkg              # most precise
  palmseg extract --label some_binary_mask.tif --out trees.gpkg
                                                          # mask alone, no heatmap

  # (c) end-to-end
  palmseg segment-and-extract --model segformer_b5_ms --checkpoint w/seg_b5_ms.pth \
                  --input scene.tif --out out/
"""
from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')


def _resolve_config(args):
    if getattr(args, 'config', None):
        return args.config
    from palmseg.weights_manifest import resolve_config, MODELS
    if getattr(args, 'model', None) in MODELS:
        return resolve_config(args.model)
    raise SystemExit('Provide --config, or --model from the manifest '
                     '(see `palmseg download --list`).')


# --- utilities --------------------------------------------------------------
def _cmd_doctor(args):
    from palmseg.doctor import run
    raise SystemExit(run())


def _cmd_inspect(args):
    from palmseg.loader import inspect_checkpoint
    s = inspect_checkpoint(args.checkpoint)
    print(f'checkpoint : {args.checkpoint}')
    print(f'in_channels: {s["in_channels"]}')
    print(f'num_classes: {s["num_classes"]}')
    print(f'modality   : {s["modality"]}')


def _cmd_selftest(args):
    import numpy as np
    from palmseg.loader import load_palm_model
    from mmseg.apis import inference_model
    model = load_palm_model(args.checkpoint, _resolve_config(args),
                            device=args.device)
    info = model.cfg_info
    nb = info['in_channels'] or 3
    tile = (np.random.rand(args.tile, args.tile, nb) * 255).astype('float32')
    res = inference_model(model, tile)
    lbl = res.pred_sem_seg.data[0].cpu().numpy()
    print(f'selftest OK: modality={info["modality"]} bands={nb} '
          f'classes={info["num_classes"]} label_shape={tuple(lbl.shape)} '
          f'unique_labels={np.unique(lbl).tolist()}')


# --- (a) segment ------------------------------------------------------------
def _require_path(path, kind, is_dir=False):
    import os.path as osp
    if path and not (osp.isdir(path) if is_dir else osp.isfile(path)):
        raise SystemExit(f'{kind} not found: {path}')


def _cmd_segment(args):
    if not args.input and not args.input_dir:
        raise SystemExit('Provide --input <file> or --input-dir <folder>.')
    # Fail fast on bad paths BEFORE the (slow) model load.
    _require_path(args.input, 'Input raster')
    _require_path(args.input_dir, 'Input folder', is_dir=True)
    _require_path(args.checkpoint, 'Checkpoint')
    from palmseg.loader import load_palm_model
    model = load_palm_model(args.checkpoint, _resolve_config(args),
                            device=args.device)
    if args.input_dir:
        from palmseg.batch import batch_segment
        res = batch_segment(model, args.input_dir, args.out,
                            pattern=args.pattern, resume=not args.no_resume,
                            tile_size=args.tile, overlap=args.overlap,
                            palm_class=args.palm_class,
                            write_heatmap=not args.no_heatmap,
                            summary_csv=args.summary_csv)
        print('batch segment:\n' + res.summary())
        return
    from palmseg.pipeline import segment
    out = segment(model, args.input, args.out, tile_size=args.tile,
                  overlap=args.overlap, palm_class=args.palm_class,
                  write_heatmap=not args.no_heatmap)
    print(f'segment: {out.tiles} tiles | label={out.label_path} '
          f'heatmap={out.prob_path}')


# --- (b) extract ------------------------------------------------------------
def _cmd_extract(args):
    if not args.label and not args.input_dir:
        raise SystemExit('Provide --label <file> or --input-dir <folder>.')
    _require_path(args.label, 'Label raster')
    _require_path(args.heatmap, 'Heatmap raster')
    _require_path(args.input_dir, 'Input folder', is_dir=True)
    if args.input_dir:
        from palmseg.batch import batch_extract
        res = batch_extract(args.input_dir, args.out, heatmap_dir=args.heatmap_dir,
                            pattern=args.pattern, resume=not args.no_resume,
                            palm_class=args.palm_class, method=args.method,
                            geometry=args.geometry, blur_ksize=args.blur,
                            prob_threshold=args.prob_thr, min_area_px=args.min_area,
                            min_circularity=args.min_circularity,
                            min_seed_distance=args.min_seed_distance,
                            dedup_iou=args.dedup_iou, summary_csv=args.summary_csv)
        print('batch extract:\n' + res.summary())
        return
    from palmseg.pipeline import extract
    out = extract(args.label, args.out, heatmap_path=args.heatmap,
                  palm_class=args.palm_class, method=args.method,
                  geometry=args.geometry, blur_ksize=args.blur,
                  prob_threshold=args.prob_thr, min_area_px=args.min_area,
                  min_circularity=args.min_circularity,
                  min_seed_distance=args.min_seed_distance,
                  dedup_iou=args.dedup_iou,
                  write_instance_raster=args.instance_raster)
    msg = (f'extract: {out.count} trees | method={out.method} '
           f'heatmap={"yes" if out.heatmap_used else "no (mask-only)"} '
           f'-> {out.vector_path}')
    print(msg)


# --- (c) segment-and-extract ------------------------------------------------
def _cmd_seg_extract(args):
    if not args.input and not args.input_dir:
        raise SystemExit('Provide --input <file> or --input-dir <folder>.')
    _require_path(args.input, 'Input raster')
    _require_path(args.input_dir, 'Input folder', is_dir=True)
    _require_path(args.checkpoint, 'Checkpoint')
    from palmseg.loader import load_palm_model
    model = load_palm_model(args.checkpoint, _resolve_config(args),
                            device=args.device)
    if args.input_dir:
        from palmseg.batch import batch_segment_and_extract
        res = batch_segment_and_extract(
            model, args.input_dir, args.out, pattern=args.pattern,
            resume=not args.no_resume, tile_size=args.tile, overlap=args.overlap,
            palm_class=args.palm_class, method=args.method,
            geometry=args.geometry, prob_threshold=args.prob_thr,
            min_area_px=args.min_area, min_circularity=args.min_circularity,
            min_seed_distance=args.min_seed_distance, summary_csv=args.summary_csv)
        print('batch segment-and-extract:\n' + res.summary())
        return
    from palmseg.pipeline import segment_and_extract
    seg, ext = segment_and_extract(
        model, args.input, args.out, tile_size=args.tile, overlap=args.overlap,
        palm_class=args.palm_class, method=args.method, geometry=args.geometry,
        prob_threshold=args.prob_thr, min_area_px=args.min_area,
        min_circularity=args.min_circularity,
        min_seed_distance=args.min_seed_distance,
        dedup_iou=args.dedup_iou)
    print(f'segment-and-extract: {seg.tiles} tiles -> {ext.count} trees '
          f'-> {ext.vector_path}')


def _cmd_adapt_stem(args):
    from palmseg.tools.adapt_input_stem import adapt_checkpoint, WV3_8BAND_RGB_MAP
    bmap = WV3_8BAND_RGB_MAP if (args.strategy == 'band_map' and
                                 args.new_in == 8) else None
    adapt_checkpoint(args.in_path, args.out_path, args.new_in, args.strategy,
                     args.stem_key, bmap, args.scale)


def _cmd_prep_check(args):
    from palmseg.tools.prep_check import check_dataset
    problems = check_dataset(args.data_root, args.modality, args.num_classes,
                             args.expected_bands)
    if not problems:
        print('prep-check: OK')
        return
    print(f'prep-check found {len(problems)} issue(s):')
    for p in problems:
        print(f'  - {p}')
    sys.exit(1)


def _cmd_download(args):
    from palmseg.weights_manifest import download_checkpoint, MODELS
    if args.list:
        for k, v in MODELS.items():
            tag = 'heatmap' if v.heatmap_capable else 'mask-only'
            print(f'  {k:24s} {v.modality:3s} {v.family:14s} [{tag}]')
        return
    if not args.all and not args.model_id:
        raise SystemExit('Provide a model id, --all, or --list '
                         '(see `palmseg download --list`).')
    ids = list(MODELS) if args.all else [args.model_id]
    for mid in ids:
        print(f'{mid} -> {download_checkpoint(mid, args.cache_dir, args.repo_id)}')


def _add_model_args(s, device=True):
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--config'); s.add_argument('--model')
    if device:
        s.add_argument('--device', default='auto',
                       help="'auto' (cuda if available, else cpu), 'cpu', 'cuda:N'")


def _add_seg_args(s):
    s.add_argument('--input', help='single raster (mode: one file)')
    s.add_argument('--input-dir', help='folder of rasters (batch mode)')
    s.add_argument('--pattern', default=None,
                   help='glob within --input-dir, e.g. "*.tif"')
    s.add_argument('--no-resume', action='store_true',
                   help='reprocess files even if outputs exist')
    s.add_argument('--summary-csv', default=None,
                   help='write a per-file batch summary CSV')
    s.add_argument('--out', default='out')
    s.add_argument('--tile', type=int, default=512)
    s.add_argument('--overlap', type=int, default=128)
    s.add_argument('--palm-class', type=int, default=1)


def _add_extract_tuning(s):
    s.add_argument('--method', choices=['watershed', 'voronoi'], default='watershed')
    s.add_argument('--geometry', choices=['circle', 'polygon'], default='circle')
    s.add_argument('--blur', type=int, default=5)
    s.add_argument('--prob-thr', type=float, default=0.5)
    s.add_argument('--min-area', type=int, default=9)
    s.add_argument('--min-circularity', type=float, default=0.0)
    s.add_argument('--min-seed-distance', type=int, default=0)
    s.add_argument('--dedup-iou', type=float, default=0.0,
                   help='merge overlapping crowns with IoU >= this (0=off)')


def build_parser():
    p = argparse.ArgumentParser(
        prog='palmseg', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-v', '--verbose', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('doctor', help='diagnose install (GDAL, CUDA, MM stack)')
    s.set_defaults(func=_cmd_doctor)

    s = sub.add_parser('inspect', help='show checkpoint channels/classes/modality')
    s.add_argument('checkpoint'); s.set_defaults(func=_cmd_inspect)

    s = sub.add_parser('selftest', help='sanity-check a model on a synthetic tile')
    _add_model_args(s); s.add_argument('--tile', type=int, default=512)
    s.set_defaults(func=_cmd_selftest)

    # (a)
    s = sub.add_parser('segment', help='(a) model -> label raster (+ heatmap)')
    _add_model_args(s); _add_seg_args(s)
    s.add_argument('--no-heatmap', action='store_true',
                   help='write only the class-label raster')
    s.set_defaults(func=_cmd_segment)

    # (b)
    s = sub.add_parser('extract', help='(b) existing raster(s) -> tree polygons')
    s.add_argument('--label', help='single label/mask raster (one file)')
    s.add_argument('--heatmap', default=None,
                   help='OPTIONAL palm-probability raster; omit for mask-only')
    s.add_argument('--input-dir', help='folder of label rasters (batch mode)')
    s.add_argument('--heatmap-dir', default=None,
                   help='folder of heatmaps matched by name (batch mode)')
    s.add_argument('--pattern', default='*_label.tif',
                   help='glob for label rasters in --input-dir')
    s.add_argument('--no-resume', action='store_true')
    s.add_argument('--summary-csv', default=None)
    s.add_argument('--out', default='trees.gpkg',
                   help='output file (one) or output FOLDER (batch)')
    s.add_argument('--palm-class', type=int, default=1)
    s.add_argument('--instance-raster', default=None,
                   help='also write the int32 instance-label raster')
    _add_extract_tuning(s)
    s.set_defaults(func=_cmd_extract)

    # (c)
    s = sub.add_parser('segment-and-extract', help='(c) end-to-end')
    _add_model_args(s); _add_seg_args(s)
    s.add_argument('--out-vector', default=None)
    _add_extract_tuning(s)
    s.set_defaults(func=_cmd_seg_extract)

    s = sub.add_parser('adapt-stem', help='expand 3->N channel stem (fine-tune)')
    s.add_argument('in_path'); s.add_argument('out_path')
    s.add_argument('--new-in', type=int, default=8)
    s.add_argument('--strategy', choices=['band_map', 'mean', 'zero'], default='band_map')
    s.add_argument('--stem-key', default=None); s.add_argument('--scale', action='store_true')
    s.set_defaults(func=_cmd_adapt_stem)

    s = sub.add_parser('prep-check', help='validate a dataset before fine-tuning')
    s.add_argument('--data-root', required=True)
    s.add_argument('--modality', choices=['ms', 'rgb'], default='ms')
    s.add_argument('--num-classes', type=int, default=2)
    s.add_argument('--expected-bands', type=int, default=None)
    s.set_defaults(func=_cmd_prep_check)

    s = sub.add_parser('download', help='fetch released weights from Hugging Face')
    s.add_argument('model_id', nargs='?'); s.add_argument('--all', action='store_true')
    s.add_argument('--list', action='store_true')
    s.add_argument('--cache-dir', default='weights'); s.add_argument('--repo-id', default=None)
    s.set_defaults(func=_cmd_download)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = getattr(args, 'verbose', False)
    _setup_logging(verbose)
    try:
        args.func(args)
    except SystemExit:
        raise                                   # already a clean exit
    except KeyboardInterrupt:
        print('\ninterrupted', file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, ValueError, KeyError, MemoryError,
            RuntimeError, ImportError, OSError) as exc:
        # Operational errors: print one actionable line, full traceback only
        # in verbose mode, and exit non-zero for scripts/schedulers.
        if verbose:
            logging.getLogger('palmseg').exception('command failed')
        print(f'error: {exc}', file=sys.stderr)
        if not verbose:
            print('(re-run with -v for the full traceback)', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
