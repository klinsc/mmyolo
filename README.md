# This is a forked of the MMYOLO

## 17.11.2023

- Found information about ignore regions. In MMYOLO doc, for model YOLOv5 of custom data, it does not mention exactly where or how we shall implement this feature. So I decided to treat this as new labels for the ASDR6 dataset which will be included "future" prefix for some components in the future scope of works in an electrical drawing diagram.
  - https://mmyolo.readthedocs.io/en/dev/recommended_topics/training_testing_tricks.html#consider-ignore-scenarios-to-avoid-uncertain-annotations
  - https://github.com/facebookresearch/detectron2/issues/1909
  - https://github.com/ultralytics/yolov5/issues/2720
- Enable cudnn_benchmark for single-scale training may improve the training speed. However, it is not recommended for multi-scale training. Using by `Enable cudnn_benchmark for single-scale training` in a config file.

- [YOLOv5 Study: mAP vs Batch-Size](https://github.com/ultralytics/yolov5/discussions/2452)
