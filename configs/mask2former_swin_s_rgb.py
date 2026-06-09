# Auto-generated canonical config: Mask2Former + Swin-S
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

# Mask2Former (Swin-S). Query-based head: for the toolkit's tree-counting /
# heatmap path, prefer the conv-seg models (SegFormer/Swin/ViT/UniFormer).
# This config is provided for inference-to-mask reproduction.
backbone_norm_cfg = dict(type='LN', requires_grad=True)
num_things_classes = num_classes
num_stuff_classes = 0
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor', size=crop_size,
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_val=0, seg_pad_val=255),
    backbone=dict(
        type='SwinTransformer', pretrain_img_size=224, in_channels=3,
        embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4,
        depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None,
        patch_norm=True, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.3, use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'), norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='Mask2FormerHead', in_channels=[96, 192, 384, 768],
        strides=[4, 8, 16, 32], feat_channels=256, out_channels=256,
        num_classes=num_classes, num_queries=100, num_transformer_feat_level=3,
        align_corners=False),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
