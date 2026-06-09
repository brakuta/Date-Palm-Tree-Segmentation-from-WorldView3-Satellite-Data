# 100k iteration schedule used for the released models.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-05, betas=(0.9, 0.999),
                   weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(
        pos_block=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        head=dict(lr_mult=10.0))))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=100000,
         by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
