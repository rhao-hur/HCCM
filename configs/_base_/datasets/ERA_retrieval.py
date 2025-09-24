rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0.1, 1.8)),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0.1, 1.8)),
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend = "pillow"),
    dict(
        type='Resize',
        scale=(384, 384),
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        magnitude_level=7),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'is_matched'],
        meta_keys=['image_id']),
]

test_pipeline = [
    dict(type='LoadImageFromFile',
         imdecode_backend = "pillow"),
    dict(
        type='Resize',
        scale=(384, 384),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_text_id', 'gt_image_id'],
        meta_keys=['image_id']),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dict(
        type='UAVDataset',
        data_root='../datasets/ERA_Dataset',
        data_prefix = dict(img_path='images'),
        test_mode = False,
        ann_file='train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dict(
        type='UAVDataset',
        data_root='../datasets/ERA_Dataset',
        data_prefix = dict(img_path='images'),
        test_mode = True,
        ann_file = "test.json",
        pipeline=test_pipeline
    ),
    sampler=dict(type='SequentialSampler', subsample_type='sequential'),
    persistent_workers=True,
)

test_dataloader = dict(
    batch_size=4,
    num_workers=16,
    dataset=dict(
        type='UAVDataset',
        data_root='../datasets/ERA_Dataset',
        data_prefix = dict(img_path='images'),
        test_mode = True,
        ann_file = "test.json",
        pipeline=test_pipeline
    ),
    sampler=dict(type='SequentialSampler', subsample_type='sequential'),
    persistent_workers=True,
)

val_evaluator = dict(type='RetrievalRecall', topk=(1, 5, 10))
test_evaluator = val_evaluator


