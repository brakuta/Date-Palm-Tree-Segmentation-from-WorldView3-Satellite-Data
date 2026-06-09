# Copyright (c) 2024 palmseg-wv3 authors. Licensed under Apache-2.0.
"""Test fixtures: stub the MM stack so tests run without MMSegmentation.

The toolkit's runtime modules import `mmseg.registry` / `mmcv.transforms` at
module load to register custom classes. In a CI/test environment without the
(heavy, CUDA-bound) MM stack installed, we provide minimal stubs so the pure
logic (postprocess, vectorise, loader helpers, doctor, batch) is importable and
testable. Real inference still requires the genuine MM stack.
"""
import sys
import types


def _ensure_stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


def pytest_configure(config):
    try:
        import mmseg  # noqa: F401
        return  # real MM stack present; do not stub
    except ImportError:
        pass

    _reg = type('R', (), {
        'register_module': staticmethod(lambda *a, **k: (lambda c: c))})()
    _ensure_stub('mmseg')
    _ensure_stub('mmseg.registry', TRANSFORMS=_reg, DATASETS=_reg)
    _ensure_stub('mmcv')
    _ensure_stub('mmcv.transforms', BaseTransform=object)
