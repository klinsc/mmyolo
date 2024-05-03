import json
import logging
import os
from argparse import ArgumentParser

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON file: %s", file_path)
        raise


def calculate_f1_score(gt, pred):
    """Calculate F1-score."""
    mlb = MultiLabelBinarizer()
    mlb.fit([range(0, 80)])  # Assuming category IDs range from 0 to 79
    gt_binary = mlb.transform(gt)
    pred_binary = mlb.transform(pred)
    f1 = f1_score(gt_binary, pred_binary, average="micro")
    return f1


def evaluate_results(result_file, ann_file, out_dir):
    """Evaluate object detection results."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load COCO dataset
    coco = COCO(ann_file)
    coco_result = coco.loadRes(result_file)

    # Evaluate using COCOeval
    coco_eval = COCOeval(coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Load image IDs
    img_ids = sorted(coco.getImgIds())
    coco_eval.params.imgIds = img_ids

    # Calculate F1-score
    gt = {}
    pred = {}
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt[img_id] = [ann["category_id"] for ann in anns]

        ann_ids = coco_result.getAnnIds(imgIds=img_id)
        anns = coco_result.loadAnns(ann_ids)
        pred[img_id] = [ann["category_id"] for ann in anns]

    f1 = calculate_f1_score(gt.values(), pred.values())

    logger.info("F1-score: %f", f1)
    return coco_eval, f1


def setup_logging(log_file=None, log_level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def main():
    parser = ArgumentParser(
        description="Calculate mAP and F1-score for COCO format dataset"
    )
    parser.add_argument("result", help="result file (json format) path")
    parser.add_argument("out_dir", help="dir to save analyze result images")
    parser.add_argument(
        "--ann",
        default="data/coco/annotations/instances_val2017.json",
        help="annotation file path",
    )
    parser.add_argument("--log_file", help="file to save logs")
    args = parser.parse_args()

    setup_logging(args.log_file)

    try:
        coco_eval, f1 = evaluate_results(args.result, args.ann, args.out_dir)
        # You can use coco_eval object for further analysis if needed
    except Exception as e:
        logger.exception("An error occurred during evaluation: %s", str(e))


if __name__ == "__main__":
    main()
