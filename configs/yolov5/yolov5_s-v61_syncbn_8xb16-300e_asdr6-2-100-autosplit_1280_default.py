_base_ = ["../_base_/default_runtime.py", "../_base_/det_p5_tta.py"]

# ========================modified parameters======================
data_root = "data/asdr6_2_100_autosplit/"

train_ann_file = "annotations/train.json"
train_data_prefix = "images/"

val_ann_file = "annotations/val.json"
val_data_prefix = "images/"

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
num_classes = len(class_name)  # Number of classes for classification
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

max_epochs = 300
train_batch_size_per_gpu = 8  # 16/12 does not fit in 15GB
train_num_workers = 4  # 8/6 does not fit in 15GB
load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_215044-58865c19.pth"

# ========================Frequently modified parameters======================
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01
max_epochs = 300  # Maximum training epochs

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
img_scale = (1280, 1280)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = "YOLOv5CocoDataset"
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type="BatchShapePolicy",
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5,
)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.4]
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type="YOLODetector",
    data_preprocessor=dict(
        type="mmdet.DetDataPreprocessor",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="YOLOv5CSPDarknet",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    neck=dict(
        type="YOLOv5PAFPN",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="YOLOv5Head",
        head_module=dict(
            type="YOLOv5HeadModule",
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=dict(
            type="mmdet.YOLOAnchorGenerator", base_sizes=anchors, strides=strides
        ),
        # scaled based on number of detection layers
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers),
        ),
        loss_bbox=dict(
            type="IoULoss",
            iou_mode="ciou",
            bbox_format="xywh",
            eps=1e-7,
            reduction="mean",
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True,
        ),
        loss_obj=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=loss_obj_weight
            * ((img_scale[0] / 640) ** 2 * 3 / num_det_layers),
        ),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
    ),
    test_cfg=model_test_cfg,
)

albu_train_transforms = [
    dict(type="Blur", p=0.01),
    dict(type="MedianBlur", p=0.01),
    dict(type="ToGray", p=0.01),
    dict(type="CLAHE", p=0.01),
]

pre_transform = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(
        type="Mosaic", img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform
    ),
    dict(
        type="YOLOv5RandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
    ),
    dict(
        type="mmdet.Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
    ),
    dict(type="YOLOv5HSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
        ),
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

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
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg,
    ),
)

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu,
    ),
    constructor="YOLOv5OptimizerConstructor",
)

default_hooks = dict(
    param_scheduler=dict(
        type="YOLOv5ParamSchedulerHook",
        scheduler_type="linear",
        lr_factor=lr_factor,
        max_epochs=max_epochs,
    ),
    checkpoint=dict(
        type="CheckpointHook",
        interval=save_checkpoint_intervals,
        save_best="auto",
        max_keep_ckpts=max_keep_ckpts,
    ),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49,
    )
]

val_evaluator = dict(
    type="mmdet.CocoMetric",
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric="bbox",
)
test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# visualization config
visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="WandbVisBackend"),
        dict(type="ClearMLVisBackend"),
    ]
)

tta_img_scales = [(1280, 1280), (1024, 1024), (1536, 1536)]
