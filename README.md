# This is a forked of the MMYOLO

## 17.11.2023

- Found information about ignore regions. In MMYOLO doc, for model YOLOv5 of custom data, it does not mention exactly where or how we shall implement this feature. So I decided to treat this as new labels for the ASDR6 dataset which will be included "future" prefix for some components in the future scope of works in an electrical drawing diagram.
  - https://mmyolo.readthedocs.io/en/dev/recommended_topics/training_testing_tricks.html#consider-ignore-scenarios-to-avoid-uncertain-annotations
  - https://github.com/facebookresearch/detectron2/issues/1909
  - https://github.com/ultralytics/yolov5/issues/2720
- Enable cudnn_benchmark for single-scale training may improve the training speed. However, it is not recommended for multi-scale training. Using by `Enable cudnn_benchmark for single-scale training` in a config file.

- [YOLOv5 Study: mAP vs Batch-Size](https://github.com/ultralytics/yolov5/discussions/2452)

## 20.11.2023

- After experiment with tuning hyperparameters, for example epoch number, image_scale(or image_size of Ultralytics's) and optimized anchor, we got 8 different results.
- We could only train at batch_size at 8 and worker at 4. Unlike in Ultralytics's YOLOv5 which we can use batch_size at 12. That might be because the model config is different.
- The epoch numbers are 150 and 300, we use 150 to test if the 300 is overfitting or not.
- From the experiment, the bigger image_scale, the better mAP. We have used image_scale at 1280 and 640 for the experiment. The mAP of 1280 is better than 640. However, the training time of 1280 is longer than 640. Usually aroung 50 minutes longer.
- The optimized anchor is a bit better than the default anchor. The mAP of optimized anchor is better than default anchor around 3%.
