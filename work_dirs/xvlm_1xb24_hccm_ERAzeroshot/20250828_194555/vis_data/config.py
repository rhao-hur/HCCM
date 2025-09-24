default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/rh/Files/MS/HCCM/pretrain/xvlm_1xb24_hccm_epoch_6.pth'
log_level = 'INFO'
model = dict(
    bbox_head=dict(hidden_size=768, type='XVLM_BOXHead'),
    fast_match=True,
    init_cfg=dict(
        checkpoint='pretrain/16m_base_model_state_step_199999_(xvlm2mmcv).pth',
        type='Pretrained'),
    itc_head=dict(
        alpha=0.4,
        embed_dim=256,
        queue_size=57600,
        type='XVLM_ITC_MCD',
        use_distill=True),
    itm_head=dict(
        cal_acc=True, hidden_size=768, type='XVLM_ITMHead', with_pooler=False),
    max_tokens=90,
    text_encoder=dict(
        med_config=dict(
            add_cross_attention=True,
            architectures=[
                'BertForMaskedLM',
            ],
            attention_probs_dropout_prob=0.1,
            encoder_width=1024,
            fusion_layer=6,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            type_vocab_size=2,
            vocab_size=30522),
        type='XVLM_XBert'),
    text_proj=dict(in_features=768, out_features=256, type='Linear'),
    tokenizer_path='pretrain/bert-base-uncased',
    topk=128,
    train_max_words=90,
    type='XVLMRetrieval_hccm',
    val_max_words=90,
    vision_encoder=dict(
        ape=False,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dim=128,
        img_size=384,
        in_chans=3,
        mlp_ratio=4.0,
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        patch_norm=True,
        patch_size=4,
        qkv_bias=True,
        type='XVLM_SwinTransformer',
        use_checkpoint=False,
        window_size=12),
    vision_proj=dict(in_features=1024, out_features=256, type='Linear'),
    w_itc=0.25,
    w_itc_entis=0.25,
    w_itm=1,
    w_itm_entis=0.5)
optim_wrapper = dict(
    optimizer=dict(lr=3e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        by_epoch=False,
        end=1000,
        end_factor=1,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end_factor=1e-10,
        start_factor=1,
        type='LinearLR'),
]
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0.1,
            1.8,
        ),
        type='Brightness'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0.1,
            1.8,
        ),
        type='Sharpness'),
]
randomness = dict(deterministic=False, seed=1)
resume = False
test_cfg = dict(
    fast_datainfo=True,
    fp16=True,
    i2t=True,
    load_cpu=True,
    type='HDCRetrievalTestLoop')
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='test.json',
        data_prefix=dict(img_path='images'),
        data_root='../datasets/ERA_Dataset',
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=(
                    384,
                    384,
                ),
                type='Resize'),
            dict(keys='text', type='CleanCaption'),
            dict(
                algorithm_keys=[
                    'text',
                    'gt_text_id',
                    'gt_image_id',
                ],
                meta_keys=[
                    'image_id',
                ],
                type='PackInputs'),
        ],
        test_mode=True,
        type='UAVDataset'),
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(subsample_type='sequential', type='SequentialSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
        10,
    ), type='RetrievalRecall')
test_pipeline = [
    dict(imdecode_backend='pillow', type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=(
            384,
            384,
        ),
        type='Resize'),
    dict(keys='text', type='CleanCaption'),
    dict(
        algorithm_keys=[
            'text',
            'gt_text_id',
            'gt_image_id',
        ],
        meta_keys=[
            'image_id',
        ],
        type='PackInputs'),
]
train_cfg = dict(max_epochs=6, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='train.json',
        data_prefix=dict(img_path='images'),
        data_root='../datasets/ERA_Dataset',
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=(
                    384,
                    384,
                ),
                type='Resize'),
            dict(
                magnitude_level=7,
                num_policies=2,
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0.1,
                            1.8,
                        ),
                        type='Brightness'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0.1,
                            1.8,
                        ),
                        type='Sharpness'),
                ],
                type='RandAugment'),
            dict(keys='text', type='CleanCaption'),
            dict(
                algorithm_keys=[
                    'text',
                    'is_matched',
                ],
                meta_keys=[
                    'image_id',
                ],
                type='PackInputs'),
        ],
        test_mode=False,
        type='UAVDataset'),
    drop_last=True,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(imdecode_backend='pillow', type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=(
            384,
            384,
        ),
        type='Resize'),
    dict(
        magnitude_level=7,
        num_policies=2,
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0.1,
                    1.8,
                ),
                type='Brightness'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0.1,
                    1.8,
                ),
                type='Sharpness'),
        ],
        type='RandAugment'),
    dict(keys='text', type='CleanCaption'),
    dict(
        algorithm_keys=[
            'text',
            'is_matched',
        ],
        meta_keys=[
            'image_id',
        ],
        type='PackInputs'),
]
val_cfg = dict(
    fast_datainfo=True,
    fp16=True,
    i2t=False,
    load_cpu=True,
    type='HDCRetrievalValLoop')
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='test.json',
        data_prefix=dict(img_path='images'),
        data_root='../datasets/ERA_Dataset',
        pipeline=[
            dict(imdecode_backend='pillow', type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=(
                    384,
                    384,
                ),
                type='Resize'),
            dict(keys='text', type='CleanCaption'),
            dict(
                algorithm_keys=[
                    'text',
                    'gt_text_id',
                    'gt_image_id',
                ],
                meta_keys=[
                    'image_id',
                ],
                type='PackInputs'),
        ],
        test_mode=True,
        type='UAVDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(subsample_type='sequential', type='SequentialSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
        10,
    ), type='RetrievalRecall')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/xvlm_1xb24_hccm_ERAzeroshot'
