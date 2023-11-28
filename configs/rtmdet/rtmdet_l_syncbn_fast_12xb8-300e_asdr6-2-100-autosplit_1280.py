_base_ = ["../_base_/default_runtime.py", "../_base_/det_p5_tta.py"]

# ========================Frequently modified parameters======================
# -----data related-----
data_root = "data/asdr6_2_100_autosplit/"
# Path of train annotation file
train_ann_file = "annotations/train.json"
train_data_prefix = "images/"  # Prefix of train image path
# Path of val annotation file
val_ann_file = "annotations/val.json"
val_data_prefix = "images/"  # Prefix of val image path
# Number of classes for classification
class_name = (
    "11522_tx_dyn1",
    "11522_tx_ynyn0d1",
    "115_1way_ds_w_motor",
    "115_3ways_ds_w_motor",
    "115_breaker",
    "115_buffer",
    "115_cvt_1p",
    "115_cvt_3p",
    "115_ds",
    "115_gs",
    "115_gs_w_motor",
    "115_la",
    "115_vt_1p",
    "115_vt_3p",
    "22_breaker",
    "22_cap_bank",
    "22_ds",
    "22_ds_la_out",
    "22_gs",
    "22_ll",
    "22_vt_3p",
    "BCU",
    "DIM",
    "DPM",
    "LL",
    "MU",
    "NGR_future",
    "Q",
    "remote_io_module",
    "ss_man_mode",
    "tele_protection",
    "terminator_double",
    "terminator_single",
    "terminator_splicing_kits",
    "terminator_w_future",
    "v_m",
    "v_m_digital",
)
num_classes = len(class_name)
palette = [
    (0, 0, 0),
    (66, 0, 75),
    (120, 0, 137),
    (130, 0, 147),
    (105, 0, 156),
    (30, 0, 166),
    (0, 0, 187),
    (0, 0, 215),
    (0, 52, 221),
    (0, 119, 221),
    (0, 137, 221),
    (0, 154, 215),
    (0, 164, 187),
    (0, 170, 162),
    (0, 170, 143),
    (0, 164, 90),
    (0, 154, 15),
    (0, 168, 0),
    (0, 186, 0),
    (0, 205, 0),
    (0, 224, 0),
    (0, 243, 0),
    (41, 255, 0),
    (145, 255, 0),
    (203, 249, 0),
    (232, 239, 0),
    (245, 222, 0),
    (255, 204, 0),
    (255, 175, 0),
    (255, 136, 0),
    (255, 51, 0),
    (247, 0, 0),
    (228, 0, 0),
    (215, 0, 0),
    (205, 0, 0),
    (204, 90, 90),
    (204, 204, 204),
]
metainfo = dict(classes=class_name, palette=palette)


# Batch size of a single GPU during training
train_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 12
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.004
max_epochs = 50  # Maximum training epochs
# Change train_pipeline for final 20 epochs (stage 2)
num_epochs_stage2 = 20

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type="nms", iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# ratio range for random resize
random_resize_ratio_range = (0.1, 2.0)
# Cached images number in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20
# Dataset type, this will be used to define the dataset
dataset_type = "YOLOv5CocoDataset"
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 16

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type="BatchShapePolicy",
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5,
)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0
# Strides of multi-scale prior box
strides = [8, 16, 32]

norm_cfg = dict(type="BN")  # Normalization config

# -----train val related-----
lr_start_factor = 1.0e-5
dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss
weight_decay = 0.05

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type="YOLODetector",
    data_preprocessor=dict(
        type="YOLOv5DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="RTMDetHead",
        head_module=dict(
            type="RTMDetSepBNHeadModule",
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="SiLU", inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides,
        ),
        prior_generator=dict(
            type="mmdet.MlvlPointGenerator", offset=0, strides=strides
        ),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="mmdet.QualityFocalLoss",
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight,
        ),
        loss_bbox=dict(type="mmdet.GIoULoss", loss_weight=loss_bbox_weight),
    ),
    train_cfg=dict(
        assigner=dict(
            type="BatchDynamicSoftLabelAssigner",
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type="mmdet.BboxOverlaps2D"),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=model_test_cfg,
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Mosaic",
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0,
    ),
    dict(
        type="mmdet.RandomResize",
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type="mmdet.Resize",
        keep_ratio=True,
    ),
    dict(type="mmdet.RandomCrop", crop_size=img_scale),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    dict(type="mmdet.Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="YOLOv5MixUp", use_cached=True, max_cached_images=mixup_max_cached_images
    ),
    dict(type="mmdet.PackDetInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="mmdet.RandomResize",
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type="mmdet.Resize",
        keep_ratio=True,
    ),
    dict(type="mmdet.RandomCrop", crop_size=img_scale),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    dict(type="mmdet.Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type="mmdet.PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="YOLOv5KeepRatioResize", scale=img_scale),
    dict(
        type="LetterResize",
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114),
    ),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "pad_param",
        ),
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type="yolov5_collate"),
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=metainfo,
    ),
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = dict(
    type="mmdet.CocoMetric",
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric="bbox",
)
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(
        type="LinearLR", start_factor=lr_start_factor, by_epoch=False, begin=0, end=1000
    ),
    dict(
        # use cosine lr from 150 to 300 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",  # Hook to save model checkpoint on specific intervals
        interval=50,  # Save model checkpoint every 10 epochs.
        max_keep_ckpts=100,  # The maximum checkpoints to keep.
    ),
    logger=dict(type="LoggerHook", interval=5),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2,
    ),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)],
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# get env "MMYOLO_TEST"
import os

isTest = os.environ.get("MMYOLO_TEST", "false").lower() == "true"

vis_backend = None
if isTest:
    vis_backend = [dict(type="LocalVisBackend")]
else:
    vis_backend = [
        dict(type="LocalVisBackend"),
        dict(type="WandbVisBackend", init_kwargs=dict(project="mmyolo-tools")),
        dict(type="ClearMLVisBackend", init_kwargs=dict(project_name="mmyolo-tools")),
    ]

# visualization config
visualizer = dict(
    vis_backends=vis_backend,
)
