# Auto-generated canonical config: UniFormer-Base + FPN
# Modality: multispectral (8-band). num_classes/in_channels are reconciled to the
# checkpoint at load time by palmseg.loader; the values here are the training
# defaults. Edit num_classes for N-class fine-tuning.
_base_ = [
    '_base_/palm_dataset_ms.py',
    '_base_/default_runtime.py',
    '_base_/schedule_100k.py',
]
num_classes = 2
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

# NOTE: requires the UniFormer backbone module registered in MMSeg
# (see docs/MODELS.md). FPN neck + FPNHead as in the released config.
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor', size=crop_size,
        mean=[0.0] * 8, std=[1.0] * 8, bgr_to_rgb=False,
        pad_val=0, seg_pad_val=255),
    backbone=dict(
        type='UniFormer', in_chans=8,
        embed_dim=[64, 128, 320, 512], layers=[5, 8, 20, 7], head_dim=64,
        mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.3, hybrid=False, windows=False, use_checkpoint=False),
    neck=dict(type='FPN', in_channels=[64, 128, 320, 512], out_channels=256,
              num_outs=4),
    decode_head=dict(
        type='FPNHead', in_channels=[256, 256, 256, 256], in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32], channels=128, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
