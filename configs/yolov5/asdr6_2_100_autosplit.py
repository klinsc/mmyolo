_base_ = "yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py"

# ========================Frequently modified parameters======================
# -----data related-----
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

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)],  # P5/32
]

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth"  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
    ),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/"),
    )
)

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best="auto"),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type="LoggerHook", interval=5),
)
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
visualizer = dict(
    vis_backends=[dict(type="LocalVisBackend"), dict(type="WandbVisBackend")]
)  # noqa
