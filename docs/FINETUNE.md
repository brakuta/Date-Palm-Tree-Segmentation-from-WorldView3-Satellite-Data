# Fine-tuning guide

You can fine-tune any released model on new imagery, for the same 2 classes
(background / palm) or for **N classes** (e.g. adding ghaf or acacia). The
process is the same; only the class count and your annotations change.

## 1. Prepare tiles

Use the data-prep tiler to turn georeferenced mosaics + polygon annotations
into MMSeg-style `img_dir/` + `ann_dir/` tiles with train/val/test splits:

```bash
# edit the CONFIGURATION block (mosaic paths, CLASS_MAP, splits) then:
python -m palmseg.tools.tile_pipeline
```

Key settings, by modality (see docs/PREPROCESSING.md for why):

- **Multispectral (8-band):** `APPLY_CONTRAST_STRETCH = False`, `BANDS = None`
  (keep all 8 bands), tiles written as multi-band `.tif`.
- **RGB:** `APPLY_CONTRAST_STRETCH = True` is fine, `BANDS = [1, 2, 3]` to drop
  an alpha channel.

For N classes, set e.g. `CLASS_MAP = {"DatePalm": 1, "Ghaf": 2}` (background is
always 0). The tiler writes a `class_palette.json` recording the mapping.

## 2. Validate the prepared dataset

```bash
palmseg prep-check --data-root data/palm_ms --modality ms --num-classes 2
```

This checks image/mask pairing, single-band index masks, class-value range, and
band-count vs modality. Fix any reported issue before training.

## 3. Prepare a starting checkpoint

Two options:

- **Fine-tune from a released palm model** (recommended when your task is
  similar). No stem surgery needed — the checkpoint already has the right input
  channels. Point `load_from` at it.
- **Start from ImageNet weights for a new band count.** Expand the 3-channel
  stem to N channels first:

  ```bash
  palmseg adapt-stem mit_b5_ade20k.pth mit_b5_8band.pth --new-in 8
  # strategies: band_map (default, places RGB kernels at WV-3 indices),
  #             mean (original method; add --scale to correct magnitude),
  #             zero (extra bands learned from scratch)
  ```

## 4. Write a fine-tune config

Copy the matching config and override the four things people actually change:

```python
# configs/finetune_palm_ms.py
_base_ = ['segformer_b5_ms.py']

# (a) point at your data
data_root = 'data/palm_ms'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader   = dict(dataset=dict(data_root=data_root))
test_dataloader  = val_dataloader

# (b) class count (raise for N-class) + class names
num_classes = 2
model = dict(decode_head=dict(num_classes=num_classes))
# UPerNet/ViT models also have an auxiliary_head -> set its num_classes too.

# (c) start from a released checkpoint (or your adapted ImageNet stem)
load_from = 'weights/segformer_b5_ms.pth'

# (d) shorter schedule + lower LR for fine-tuning
train_cfg = dict(max_iters=20000, val_interval=2000)
optim_wrapper = dict(optimizer=dict(lr=1e-5))
```

For N classes, also override the dataset `metainfo`:

```python
train_dataloader = dict(dataset=dict(
    metainfo=dict(classes=('background', 'palm', 'ghaf'),
                  palette=[[0, 0, 0], [255, 0, 37], [0, 128, 0]])))
```

When changing the class count relative to the checkpoint, the head will not
load (shape mismatch on `conv_seg`); that is expected — the backbone loads and
the head is trained fresh. Pass the checkpoint via `load_from` (not `resume`).

## 5. Train

```bash
python -m mmengine.tools.train configs/finetune_palm_ms.py
# or use mmsegmentation's tools/train.py from your MMSeg install:
python tools/train.py configs/finetune_palm_ms.py
```

## 6. Use the fine-tuned model

Inference and tree-counting are identical to the released models:

```bash
palmseg infer --config configs/finetune_palm_ms.py \
              --checkpoint work_dirs/finetune_palm_ms/iter_20000.pth \
              --input new_scene.tif --out out/
palmseg count-trees --label out/new_scene_label.tif \
              --prob out/new_scene_prob.tif --out out/new_scene_trees.gpkg
```
