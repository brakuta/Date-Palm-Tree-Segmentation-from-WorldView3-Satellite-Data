# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Environment diagnostics.

Checks the components that commonly fail to install and prints a fix for each:
the raster backend (rasterio/GDAL), PyTorch and CUDA (including the install
command matching the detected CUDA), the MM stack, and this package. Exits 0 if
the requirements for inference are met, 1 otherwise. Extraction-only use does
not need PyTorch or MMSegmentation, so those are reported as warnings.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import List


GREEN, RED, YELLOW, RESET = '\033[92m', '\033[91m', '\033[93m', '\033[0m'


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ''
    fix: str = ''
    required: bool = True


def _try_import(mod: str):
    try:
        return importlib.import_module(mod), None
    except Exception as exc:                       # noqa: BLE001
        return None, exc


def _check_raster_backend() -> Check:
    rio, _ = _try_import('rasterio')
    if rio is not None:
        gdal_ver = getattr(rio, '__gdal_version__', '?')
        return Check('raster backend (rasterio)', True,
                     f'rasterio {rio.__version__}, bundled GDAL {gdal_ver}')
    osgeo, _ = _try_import('osgeo.gdal')
    if osgeo is not None:
        return Check('raster backend (osgeo.gdal)', True, 'osgeo.gdal present')
    return Check(
        'raster backend', False,
        'neither rasterio nor osgeo.gdal importable',
        fix='pip install rasterio   # bundles GDAL; no separate GDAL needed\n'
            '     (or) conda install -c conda-forge rasterio')


def _check_geospatial() -> List[Check]:
    out = []
    for mod, pip in [('geopandas', 'geopandas'), ('shapely', 'shapely'),
                     ('skimage', 'scikit-image'), ('scipy', 'scipy'),
                     ('cv2', 'opencv-python-headless')]:
        m, err = _try_import(mod)
        out.append(Check(
            f'{mod}', m is not None,
            getattr(m, '__version__', '') if m else str(err),
            fix=f'pip install {pip}'))
    return out


def _cuda_pytorch_check() -> List[Check]:
    out = []
    torch, err = _try_import('torch')
    if torch is None:
        out.append(Check(
            'pytorch', False, str(err), required=False,
            fix='Install the build matching your CUDA. Find your CUDA with '
                '`nvidia-smi` (top-right), then from pytorch.org, e.g. CUDA '
                '12.1:\n     pip install torch torchvision '
                '--index-url https://download.pytorch.org/whl/cu121'))
        return out
    out.append(Check('pytorch', True, f'torch {torch.__version__}',
                     required=False))
    cuda_build = getattr(torch.version, 'cuda', None)
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:                          # noqa: BLE001
            name = 'GPU'
        out.append(Check('CUDA usable by PyTorch', True,
                         f'CUDA build {cuda_build}, device: {name}',
                         required=False))
    else:
        # PyTorch installed but no usable GPU: most often a CPU-only wheel or a
        # driver/runtime mismatch.
        if cuda_build is None:
            detail = 'torch is a CPU-only build (torch.version.cuda is None)'
            fix = ('Reinstall a CUDA build matching your driver. Check CUDA via '
                   '`nvidia-smi`, then e.g. CUDA 12.1:\n'
                   '     pip uninstall -y torch torchvision\n'
                   '     pip install torch torchvision '
                   '--index-url https://download.pytorch.org/whl/cu121')
        else:
            detail = (f'torch built for CUDA {cuda_build} but no GPU is '
                      f'visible (driver/runtime mismatch or no GPU)')
            fix = ('Verify the NVIDIA driver with `nvidia-smi`. If it works, '
                   'ensure the torch CUDA build <= driver CUDA version. '
                   'Inference still runs on CPU, just slowly.')
        out.append(Check('CUDA usable by PyTorch', False, detail, fix=fix,
                         required=False))
    return out


def _check_mm_stack() -> List[Check]:
    out = []
    for mod, required, hint in [
            ('mmengine', False, 'mim install mmengine'),
            ('mmcv', False, 'mim install "mmcv>=2.1.0,<2.2.0"'),
            ('mmseg', False, 'mim install "mmsegmentation>=1.2.0"'),
            ('mmdet', False, 'mim install "mmdet>=3.0.0"  # Mask2Former only')]:
        m, err = _try_import(mod)
        ver = getattr(m, '__version__', '') if m else str(err)
        out.append(Check(mod, m is not None, ver, fix=hint, required=required))
    return out


def _check_toolkit() -> List[Check]:
    out = []
    for mod in ['palmseg.postprocess.individual_trees',
                'palmseg.postprocess.vectorize', 'palmseg.batch']:
        m, err = _try_import(mod)
        out.append(Check(mod, m is not None,
                         '' if m else str(err),
                         fix='pip install -e .  (from the repo root)'))
    return out


def run() -> int:
    print('Date Palm Segmentation (WorldView-3) - environment doctor\n')
    sections = [
        ('Raster backend (GDAL)', [_check_raster_backend()]),
        ('Geospatial / vision deps', _check_geospatial()),
        ('PyTorch + CUDA (GPU)', _cuda_pytorch_check()),
        ('MM stack (inference/training)', _check_mm_stack()),
        ('This toolkit', _check_toolkit()),
    ]
    hard_fail = False
    for title, checks in sections:
        print(f'== {title} ==')
        for c in checks:
            mark = (f'{GREEN}OK{RESET}' if c.ok
                    else (f'{RED}FAIL{RESET}' if c.required
                          else f'{YELLOW}WARN{RESET}'))
            print(f'  [{mark}] {c.name}: {c.detail}')
            if not c.ok and c.fix:
                for line in c.fix.split('\n'):
                    print(f'         fix> {line}')
            if not c.ok and c.required:
                hard_fail = True
        print()

    # Capability summary
    rio_ok = _try_import('rasterio')[0] is not None or \
        _try_import('osgeo.gdal')[0] is not None
    torch_ok = _try_import('torch')[0] is not None
    mmseg_ok = _try_import('mmseg')[0] is not None
    print('== What you can run now ==')
    print(f'  extraction-only (mode b): '
          f'{"YES" if rio_ok else "NO - install rasterio"}')
    print(f'  segmentation/inference (modes a, c): '
          f'{"YES" if (torch_ok and mmseg_ok and rio_ok) else "NO - see WARN/FAIL above"}')
    print()
    if hard_fail:
        print(f'{RED}Some required components are missing.{RESET} '
              f'Fix the FAIL items above.')
        return 1
    print(f'{GREEN}Core requirements satisfied.{RESET}')
    return 0


if __name__ == '__main__':
    sys.exit(run())
