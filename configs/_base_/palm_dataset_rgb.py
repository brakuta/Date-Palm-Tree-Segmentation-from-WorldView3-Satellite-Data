# Register this toolkit's custom dataset and transform with MMSegmentation
# WITHOUT modifying a cloned mmsegmentation tree. MMEngine imports these
# modules (triggering @DATASETS / @TRANSFORMS registration) before building
# anything, so `python tools/train.py <config>` from a stock mmseg clone
# works as-is. allow_failed_imports=False makes a missing palmseg install a
# clear error rather than a confusing 'type not in registry'.
custom_imports = dict(
    imports=['palmseg.datasets.palm_dataset', 'palmseg.transforms.loading'],
    allow_failed_imports=False)

# Base dataset config: RGB palm tiles.
# Contract: OpenCV BGR load -> bgr_to_rgb -> ImageNet mean/std,
# loaded via LoadImageFromFile. Tiles may be .tif (3-band) or .png/.jpg.
dataset_type = 'PalmDataset'
data_root = 'data/palm_rgb'            # <- edit
crop_size = (512, 512)
num_classes = 2

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0, seg_pad_val=255)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path='img_dir/train',
                                  seg_map_path='ann_dir/train'),
                 img_suffix='.tif', seg_map_suffix='.png',
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path='img_dir/val',
                                  seg_map_path='ann_dir/val'),
                 img_suffix='.tif', seg_map_suffix='.png',
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
