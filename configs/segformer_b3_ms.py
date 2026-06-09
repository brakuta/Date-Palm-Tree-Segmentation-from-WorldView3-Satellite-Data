# SegFormer MiT-B3. Modality: multispectral (8-band).
# num_classes / in_channels / preprocessor are reconciled to the checkpoint at
# load time by palmseg.loader; values here are the training defaults.
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
        type='MixVisionTransformer', in_channels=8, embed_dims=64,
        num_stages=4, num_layers=[3, 4, 18, 3], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead', in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3], channels=256, dropout_ratio=0.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
