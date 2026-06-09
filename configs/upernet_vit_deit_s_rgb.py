# Auto-generated canonical config: UPerNet + ViT-DeiT-S16
# Modality: RGB (3-band). num_classes/in_channels are reconciled to the
# checkpoint at load time by palmseg.loader; the values here are the training
# defaults. Edit num_classes for N-class fine-tuning.
_base_ = [
    '_base_/palm_dataset_rgb.py',
    '_base_/default_runtime.py',
    '_base_/schedule_100k.py',
]
num_classes = 2
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor', size=crop_size,
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_val=0, seg_pad_val=255),
    backbone=dict(
        type='VisionTransformer', img_size=(512, 512), in_channels=3,
        patch_size=16, embed_dims=384, num_layers=12, num_heads=6,
        mlp_ratio=4, out_indices=(2, 5, 8, 11), qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic'),
    neck=dict(type='MultiLevelNeck', in_channels=[384, 384, 384, 384],
              out_channels=384, scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead', in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512,
        dropout_ratio=0.1, num_classes=num_classes, norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead', in_channels=384, in_index=3, channels=256,
        num_convs=1, concat_input=False, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=0.4)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
