# Description: inference for asdr, take config, checkpoint, image as input, return result

import argparse

import cv2
import mmcv
from matplotlib import pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def args_parser():
    parser = argparse.ArgumentParser(description="inference for asdr")
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--checkpoint", type=str, help="checkpoint file path")
    parser.add_argument("--image", type=str, help="image file path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device used for inference",
        required=False,
    )
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    image = mmcv.imread(args.image)
    result = inference_detector(
        model,
        image,
    )

    # init the visualizer(execute this block only once)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # show the results
    visualizer.add_datasample(
        "result",
        image,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file="outputs/result.png",  # optionally, write to output file
    )
    visualizer.show()


if __name__ == "__main__":
    main()
