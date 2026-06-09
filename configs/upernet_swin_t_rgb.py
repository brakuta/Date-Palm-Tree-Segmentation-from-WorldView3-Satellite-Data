# UPerNet + Swin-Tiny (RGB (3-band)). num_classes/in_channels are reconciled to the
# checkpoint at load time by palmseg.loader.
custom_imports = dict(
    imports=['palmseg.datasets.palm_dataset', 'palmseg.transforms.loading'],
    allow_failed_imports=False)

_base_ = [
    '_base_/palm_dataset_rgb.py',
    '_base_/default_runtime.py',
    '_base_/schedule_100k.py',
]
num_classes = 2
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor', size=crop_size,
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_val=0, seg_pad_val=255),
    backbone=dict(
        type='SwinTransformer', pretrain_img_size=224, in_channels=3,
        embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None,
        patch_norm=True, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.3, use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'), norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='UPerHead', in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512,
        dropout_ratio=0.1, num_classes=num_classes, norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead', in_channels=384, in_index=2, channels=256,
        num_convs=1, concat_input=False, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=0.4)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
