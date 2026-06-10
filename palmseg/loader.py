# Copyright (c) 2024. Licensed under Apache-2.0.
"""Model loading with checkpoint introspection.

Builds an MMSegmentation model from a config and a checkpoint, reconciling the
config to the weights before loading:

  - input channels are read from the backbone stem and applied to the
    preprocessor (8-band multispectral vs 3-band RGB);
  - the class count is read from the segmentation head and written into the
    config heads;
  - the data preprocessor is set to the modality's contract (raw mean=0/std=1
    for multispectral, ImageNet statistics with bgr_to_rgb for RGB).

This avoids shape mismatches when a config carries a different ``num_classes``
than the trained checkpoint, and removes the need to hand-edit configs per
checkpoint. Use ``inspect_checkpoint`` to read these values without building a
model.
"""
from __future__ import annotations

import logging
import os.path as osp
from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:                                    # type hints only
    from mmengine.config import Config

logger = logging.getLogger('palmseg.loader')

# mmengine / mmseg and the custom-type registrations are imported lazily inside
# the functions that build models, so that checkpoint INSPECTION (channels /
# classes / modality) works on machines without the MM stack installed.


# --- preprocessing contracts (must match training exactly) ------------------
# Multispectral ("All"): raw 0-255 DN, no normalisation, no channel swap.
MS_MEAN_STD = (0.0, 1.0)
# RGB: OpenCV BGR load -> bgr_to_rgb -> ImageNet mean/std.
RGB_MEAN = [123.675, 116.28, 103.53]
RGB_STD = [58.395, 57.12, 57.375]


def _strip_state_dict(ckpt: dict) -> dict:
    """Return the parameter dict from a raw mmengine checkpoint or bare dict."""
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def _is_safetensors(path: str) -> bool:
    return path.lower().endswith('.safetensors')


def load_state_dict_any(checkpoint_path: str) -> dict:
    """Read a flat parameter dict from a ``.pth`` or ``.safetensors`` file.

    safetensors files are tensor-only (no pickle, no executable payload), which
    is why the released weights are published in that format. A ``.pth`` file is
    still accepted for local/legacy checkpoints.
    """
    if _is_safetensors(checkpoint_path):
        from safetensors.torch import load_file
        return load_file(checkpoint_path)            # already a flat dict
    # Legacy .pth: weights_only=False is needed for mmengine checkpoints whose
    # meta holds pickled objects. Released weights ship as safetensors above.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return _strip_state_dict(ckpt)


def _find_stem_in_channels(state_dict: dict) -> Optional[int]:
    """Infer model input channels from the backbone stem conv weight.

    Handles:
      * SegFormer (MixVisionTransformer): backbone.layers.0.0.projection.weight
      * Swin / Mask2Former / ViT: backbone.patch_embed.projection.weight
      * generic: the earliest 4-D backbone conv whose kernel spatial size > 1
    """
    candidates = []
    for k, v in state_dict.items():
        if not torch.is_tensor(v) or v.dim() != 4:
            continue
        if 'backbone' not in k:
            continue
        if k.endswith('projection.weight') or k.endswith('proj.weight') \
                or 'patch_embed' in k or k.endswith('stem.0.weight'):
            candidates.append((k, v))
    if not candidates:
        return None
    # Prefer patch_embed / earliest layer; in_channels is dim=1 of the conv.
    candidates.sort(key=lambda kv: (0 if 'patch_embed' in kv[0] else 1, kv[0]))
    return int(candidates[0][1].shape[1])


def _find_head_num_classes(state_dict: dict) -> Optional[int]:
    """Infer the number of classes from the decode-head classifier.

    Different heads name the final classifier differently:
      * SegformerHead / FCNHead / UPerHead: decode_head.conv_seg.weight  -> [C,...]
      * Mask2Former: decode_head.cls_embed.weight -> [num_queries? , ...]
        (Mask2Former predicts class logits per query; for semantic export the
        relevant count is the number of *thing/stuff* classes in cls_embed's
        out dim minus the void/no-object slot). We detect conv_seg first and
        only fall back to cls_embed when conv_seg is absent.
    """
    # Primary: a conv_seg classifier (the common case for the released models).
    for k, v in state_dict.items():
        if k.endswith('decode_head.conv_seg.weight') and torch.is_tensor(v):
            return int(v.shape[0])
    # Fallback: Mask2Former-style classification embedding.
    for k, v in state_dict.items():
        if k.endswith('cls_embed.weight') and torch.is_tensor(v):
            # cls_embed maps query features -> (num_classes + 1) including the
            # no-object class; subtract it for the semantic class count.
            return int(v.shape[0]) - 1
    return None


def resolve_device(device: str = 'auto') -> str:
    """Resolve 'auto' to 'cuda:0' when CUDA is available, else 'cpu'.

    Any explicit device string is returned unchanged, but requesting CUDA on a
    machine without it raises a clear error instead of a deep torch traceback.
    """
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f"device='{device}' requested but CUDA is not available. "
            "Use --device cpu (slower) or install a CUDA-enabled torch build.")
    return device


def inspect_checkpoint(checkpoint_path: str) -> dict:
    """Return a summary of what a checkpoint actually contains.

    Keys: in_channels, num_classes, modality ('ms' | 'rgb' | 'unknown').
    """
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    ckpt_sd = load_state_dict_any(checkpoint_path)
    sd = _strip_state_dict(ckpt_sd)
    in_ch = _find_stem_in_channels(sd)
    n_cls = _find_head_num_classes(sd)
    if in_ch == 3:
        modality = 'rgb'
    elif in_ch is not None and in_ch >= 4:
        modality = 'ms'
    else:
        modality = 'unknown'
    return dict(in_channels=in_ch, num_classes=n_cls, modality=modality)


def _override_num_classes(cfg: Config, num_classes: int) -> None:
    """Set num_classes on every head present in the model config."""
    model = cfg.model
    for head_key in ('decode_head', 'auxiliary_head'):
        head = model.get(head_key, None)
        if head is None:
            continue
        heads = head if isinstance(head, (list, tuple)) else [head]
        for h in heads:
            if isinstance(h, dict) and 'num_classes' in h:
                h['num_classes'] = num_classes
    # Mask2Former carries num_classes on the panoptic/decode head too.
    if 'decode_head' in model and isinstance(model['decode_head'], dict):
        if 'num_classes' in model['decode_head']:
            model['decode_head']['num_classes'] = num_classes


def _override_preprocessor(cfg: Config, modality: str, in_channels: int) -> None:
    """Force the data preprocessor to the correct per-modality contract."""
    dp = cfg.model.setdefault('data_preprocessor', {})
    dp['type'] = 'SegDataPreProcessor'
    dp.setdefault('size', (512, 512))
    dp['pad_val'] = 0
    dp['seg_pad_val'] = 255
    if modality == 'rgb':
        dp['mean'] = list(RGB_MEAN)
        dp['std'] = list(RGB_STD)
        dp['bgr_to_rgb'] = True
    else:
        # Multispectral / "All": raw values, no normalisation. mean/std are
        # per-channel zeros/ones. bgr_to_rgb is a no-op at >3 channels but we
        # keep it False to avoid any ambiguity.
        dp['mean'] = [0.0] * in_channels
        dp['std'] = [1.0] * in_channels
        dp['bgr_to_rgb'] = False


def build_inference_config(checkpoint_path: str,
                           config_path: Optional[str] = None,
                           override: bool = True) -> Tuple[Config, dict]:
    """Build an mmengine Config aligned to the checkpoint.

    Args:
        checkpoint_path: trained .pth.
        config_path: model config. Required (the architecture cannot be
            reconstructed from weights alone), but its ``num_classes``,
            ``in_channels`` and preprocessor are overridden to match the
            checkpoint when ``override`` is True.
        override: if True (default), reconcile config to checkpoint.

    Returns:
        (cfg, info) where info is the inspect_checkpoint() summary.
    """
    if config_path is None:
        raise ValueError(
            'A model config path is required to define the architecture. '
            'Pass the matching config from configs/; its num_classes / '
            'in_channels / preprocessor will be auto-corrected to the '
            'checkpoint.')
    from mmengine.config import Config
    # Registers PalmDataset and LoadSingleRSImageFromFile so Config.fromfile +
    # init_model can resolve the custom 'types'.
    from palmseg.datasets import palm_dataset    # noqa: F401
    from palmseg.transforms import loading        # noqa: F401
    info = inspect_checkpoint(checkpoint_path)
    cfg = Config.fromfile(config_path)
    cfg.work_dir = './work_dirs'

    if override:
        if info['num_classes'] is not None:
            _override_num_classes(cfg, info['num_classes'])
        if info['in_channels'] is not None:
            _override_preprocessor(cfg, info['modality'], info['in_channels'])
        # Defensive: clear any backbone init_cfg that would try to fetch an
        # ImageNet pretrain over the network at load time. The trained weights
        # already populate the backbone.
        if isinstance(cfg.model.get('backbone'), dict):
            cfg.model['backbone']['init_cfg'] = None
        cfg.model['pretrained'] = None
    return cfg, info


def _verified_load(model, state: dict, checkpoint_path: str) -> None:
    """Load a flat parameter dict into a built model and VERIFY the result.

    ``strict=False`` loading can silently leave parts of the network at random
    initialisation when the config does not match the checkpoint (e.g. a wrong
    backbone depth), producing a model that runs but predicts garbage. This
    helper compares the two key sets and tensor shapes explicitly and raises a
    diagnosable error on any real mismatch.
    """
    model_sd = model.state_dict()
    ignorable = ('num_batches_tracked',)

    missing = [k for k in model_sd
               if k not in state and not k.endswith(ignorable)]
    unexpected = [k for k in state
                  if k not in model_sd and not k.endswith(ignorable)]
    mismatched = [
        (k, tuple(state[k].shape), tuple(model_sd[k].shape))
        for k in state
        if k in model_sd and tuple(state[k].shape) != tuple(model_sd[k].shape)
    ]
    if missing or mismatched:
        def _head(items, n=5):
            shown = ', '.join(str(i) for i in items[:n])
            more = f' (+{len(items) - n} more)' if len(items) > n else ''
            return shown + more
        parts = []
        if missing:
            parts.append(f'{len(missing)} parameter(s) missing from the '
                         f'checkpoint: {_head(missing)}')
        if mismatched:
            parts.append(f'{len(mismatched)} shape mismatch(es) '
                         f'(checkpoint vs config): {_head(mismatched, 3)}')
        raise RuntimeError(
            f'Config/checkpoint mismatch loading {checkpoint_path}: '
            + '; '.join(parts)
            + '. The config architecture (backbone depths/widths) must match '
              'the run that produced the checkpoint - pass the correct '
              '--config, or fix the backbone block in the config.')
    if unexpected:
        logger.warning('%d checkpoint key(s) not used by the model '
                       '(first: %s)', len(unexpected), unexpected[0])
    model.load_state_dict(
        {k: v for k, v in state.items() if k in model_sd}, strict=False)


def load_palm_model(checkpoint_path: str,
                    config_path: Optional[str] = None,
                    device: str = 'auto',
                    override: bool = True):
    """Load a released palm model ready for inference.

    Returns an mmseg model (via mmseg.apis.init_model) with num_classes,
    input channels and preprocessing reconciled to the checkpoint. Loaded
    weights are verified against the built architecture; a config that does
    not match the checkpoint raises instead of silently producing a partially
    initialised model.

    Args:
        device: 'auto' (default) resolves to cuda:0 when available, else cpu.

    Example:
        model = load_palm_model(
            'weights/segformer_b5_ms.safetensors',
            'configs/segformer_b5_ms.py')
    """
    from mmengine.registry import init_default_scope
    from mmseg.apis import init_model

    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    device = resolve_device(device)
    init_default_scope('mmseg')
    cfg, info = build_inference_config(checkpoint_path, config_path, override)

    if _is_safetensors(checkpoint_path):
        # safetensors holds raw tensors only and cannot be passed to
        # init_model's pickle-based loader. Build the architecture first, then
        # load the parameter dict with explicit verification.
        model = init_model(cfg, None, device=device)
        state = load_state_dict_any(checkpoint_path)
        _verified_load(model, state, checkpoint_path)
        from palmseg.datasets.palm_dataset import PALM_CLASSES, PALM_PALETTE
        model.dataset_meta = dict(classes=PALM_CLASSES, palette=PALM_PALETTE)
        model.to(device).eval()
    else:
        model = init_model(cfg, checkpoint_path, device=device)
    # Stash what we detected so downstream code (heatmap class index, band
    # count) can rely on it without re-opening the checkpoint.
    model.cfg_info = info
    return model


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='Inspect a palm checkpoint (channels / classes / modality).')
    ap.add_argument('checkpoint')
    args = ap.parse_args()
    summary = inspect_checkpoint(args.checkpoint)
    print(f'checkpoint : {args.checkpoint}')
    print(f'in_channels: {summary["in_channels"]}')
    print(f'num_classes: {summary["num_classes"]}')
    print(f'modality   : {summary["modality"]}')
