# UniFormer-XS (UniFormer_Light + FPN). Modality: multispectral (8-band).
# Backbone values are taken from the training config that produced
# uniformer_xs_ms.safetensors (type='UniFormer_Light', conv_stem, pruning).
# num_classes / in_channels / preprocessor are reconciled to the checkpoint at
# load time by palmseg.loader.
#
# NOTE: UniFormer_Light is a CUSTOM backbone, not part of stock MMSegmentation.
# It must be registered in your MMSeg install for this config to build.
_base_ = [
    '_base_/palm_dataset_ms.py',
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
        mean=[0.0] * 8, std=[1.0] * 8, bgr_to_rgb=False,
        pad_val=0, seg_pad_val=255),
    backbone=dict(
        type='UniFormer_Light', in_chans=8, conv_stem=True,
        embed_dim=[64, 128, 256, 512], depth=[3, 5, 9, 3], head_dim=32,
        mlp_ratio=[3, 3, 3, 3], qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        prune_ratio=[[], [],
                     [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5]],
        trade_off=[[], [],
                   [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]]),
    neck=dict(type='FPN', in_channels=[64, 128, 256, 512], out_channels=256,
              num_outs=4),
    decode_head=dict(
        type='FPNHead', in_channels=[256, 256, 256, 256], in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32], channels=128, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
