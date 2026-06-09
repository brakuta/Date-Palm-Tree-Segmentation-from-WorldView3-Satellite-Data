# Register this toolkit's custom dataset and transform with MMSegmentation
# WITHOUT modifying a cloned mmsegmentation tree. MMEngine imports these
# modules (triggering @DATASETS / @TRANSFORMS registration) before building
# anything, so `python tools/train.py <config>` from a stock mmseg clone
# works as-is. allow_failed_imports=False makes a missing palmseg install a
# clear error rather than a confusing 'type not in registry'.
custom_imports = dict(
    imports=['palmseg.datasets.palm_dataset', 'palmseg.transforms.loading'],
    allow_failed_imports=False)

# Base dataset config: multispectral ("All", 8-band) palm tiles.
# Contract: raw DN values (no stretch), GDAL-native band order,
# data preprocessor mean=0/std=1, loaded via LoadSingleRSImageFromFile.
dataset_type = 'PalmDataset'           # registered also as 'ADE20KDataset' (legacy)
data_root = 'data/palm_ms'             # <- edit: contains img_dir/ ann_dir/
crop_size = (512, 512)
num_classes = 2                        # background + palm; raise for N-class

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0.0] * 8, std=[1.0] * 8,     # raw values; expand list for >8 bands
    bgr_to_rgb=False,
    pad_val=0, seg_pad_val=255)

train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
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
