_base_ = "yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"

# data related
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

max_epochs = 150
train_batch_size_per_gpu = 8  # 16/12 does not fit in 15GB
train_num_workers = 4  # 8/6 does not fit in 15GB
load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_215044-58865c19.pth"
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    ),
)
train_cfg = dict(max_epochs=max_epochs, val_interval=10)


# ========================modified parameters======================
img_scale = (640, 640)  # width, height
# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=64,
)
# Basic size of multi-scale prior box
anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
    [(436, 615), (739, 380), (925, 792)],  # P6/64
]
# Strides of multi-scale prior box
strides = [8, 16, 32, 64]
num_det_layers = 4  # The number of model output scales
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.25, 0.06]
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

tta_img_scales = [(1280, 1280), (1024, 1024), (1536, 1536)]
# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(arch="P6", out_indices=(2, 3, 4, 5)),
    neck=dict(in_channels=[256, 512, 768, 1024], out_channels=[256, 512, 768, 1024]),
    bbox_head=dict(
        head_module=dict(in_channels=[256, 512, 768, 1024], featmap_strides=strides),
        prior_generator=dict(base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers)
        ),
        loss_bbox=dict(loss_weight=loss_bbox_weight * (3 / num_det_layers)),
        loss_obj=dict(
            loss_weight=loss_obj_weight
            * ((img_scale[0] / 640) ** 2 * 3 / num_det_layers)
        ),
        obj_level_weights=obj_level_weights,
    ),
)

pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms

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
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/"),
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
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/"),
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator

# Config for Test Time Augmentation. (TTA)
_multiscale_resize_transforms = [
    dict(
        type="Compose",
        transforms=[
            dict(type="YOLOv5KeepRatioResize", scale=s),
            dict(
                type="LetterResize",
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114),
            ),
        ],
    )
    for s in tta_img_scales
]

tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(
        type="TestTimeAug",
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type="mmdet.RandomFlip", prob=1.0),
                dict(type="mmdet.RandomFlip", prob=0.0),
            ],
            [dict(type="mmdet.LoadAnnotations", with_bbox=True)],
            [
                dict(
                    type="mmdet.PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]

# hooks
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best="auto"),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=1000),
    logger=dict(type="LoggerHook", interval=5),
)

# visualization config
visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="WandbVisBackend"),
        dict(type="ClearMLVisBackend"),
    ]
)
