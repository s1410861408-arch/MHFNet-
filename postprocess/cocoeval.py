from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoeval import Params
from IPython.display import JSON
import numpy as np
import argparse
import logging
import json
import os


def set_category_id(input_path, output_path, target_id=1):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 读取JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化计数器
    modified_count = 0
    total_bbox = len(data)

    for item in data:
        item["image_id"] = int(item["image_id"])
        if 'category_id' in item:
            item['category_id'] = target_id
            modified_count += 1

    # 保存修改后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print(f"成功处理文件：{os.path.abspath(input_path)}")
    print(f"• 修改的annotation数量：{modified_count}")
    print(f"• 目标category_id：{target_id}")
    print(f"• 结果保存至：{os.path.abspath(output_path)}")
    print(f"• 总检测框数量：{total_bbox}")
    return total_bbox


def evaluate_metrics(cocoEval, params=None, display_summary=False):
    # Display the iouThresholds for which the evaluation took place
    if params:
        cocoEval.params = params
    print("IoU Thresholds: ", cocoEval.params.iouThrs)
    iou_lookup = {float(format(val, '.2f')): index for index, val in enumerate(cocoEval.params.iouThrs)}

    cocoEval.evaluate()  # Calculates the metrics for each class
    cocoEval.accumulate(p=params)  # Stores the values in the cocoEval's 'eval' object
    if display_summary:
        cocoEval.summarize()  # Display the metrics.

    # Extract the metrics from accumulated results.
    precision = cocoEval.eval["precision"]
    recall = cocoEval.eval["recall"]
    scores = cocoEval.eval["scores"]

    return precision, recall, scores, iou_lookup


# Print final results
def display_metrics(precision, recall, scores, iou_lookup, class_name=None, log_path='evaluation.txt'):
    # Initialize logger
    logger = logging.getLogger('eval_log')
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_path)
        logger.addHandler(handler)

    if class_name:
        logger.warning("| Class Name | IoU | mAP | F1-Score | Precision | Recall |")
        logger.warning("|------------|-----|-----|----------|-----------|--------|")
    else:
        logger.warning("| IoU | mAP | F1-Score | Precision | Recall |")
        logger.warning("|-----|-----|----------|-----------|--------|")

    for iou in iou_lookup.keys():
        precesion_iou = precision[iou_lookup[iou], :, :, 0, -1].mean(1)
        scores_iou = scores[iou_lookup[iou], :, :, 0, -1].mean(1)
        recall_iou = recall[iou_lookup[iou], :, 0, -1]
        prec = precesion_iou.mean()
        rec = recall_iou.mean()

        if class_name:
            logger.warning("|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                class_name, iou, prec * 100, scores_iou.mean(), ((2 * prec * rec) / (prec + rec)), prec, rec
            ))
            if iou == 0.5:
                gt = args.gt_eval_number
                tp = gt * rec
                precision_print = tp / det_bbox
                fp = det_bbox - tp
                fn = gt - tp
                print("Class Name: {:10s} IoU: {:2.2f} mAP: {:6.3f} F1-Score: {:6.3f} Precision: {:6.3f} Recall: {:6.3f}".format(
                   class_name, iou, prec * 100, (2 * prec * rec / (prec + rec + 1e-8)) * 100, precision_print * 100, rec * 100
                ))
                print("GT: {:6.3f} DET: {:6.3f} TP: {:6.3f} FP: {:6.3f} FN: {:6.3f}".format(
                   gt, det_bbox, tp, fp, fn
                ))

        else:
            # print("IoU: {:2.2f} mAP: {:6.3f} F1-Score: {:2.3f} Precision: {:2.2f} Recall: {:2.2f}".format(
            #    iou, prec * 100,scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec
            # ))

            logger.warning("|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|".format(
                iou, prec * 100, scores_iou.mean(), (2 * prec * rec / (prec + rec + 1e-8)), prec, rec
            ))


def _compute_and_display_metrics(args):
    coco_gt = COCO(args.gt_coco_path,)
    coco_pred = coco_gt.loadRes(args.evaluation_result_path)
    cocoEval = COCOeval(coco_gt, coco_pred, "bbox")
    # Load the default parameters for COCOEvaluation
    params = cocoEval.params

    ### Modify required parameters. Available params are:
     #imgIds          - [all],
     #catIds          - [all],
     #iouThrs         - [.5:.05:.95],
     #areaRng,maxDets - [1 10 100],
     #iouType         - ['bbox'],useCats
     # eg. param.iouType = 'bbox'
     #params.iouThrs = np.linspace(.5, .9, int(np.round((.9 - .5) / .1)) + 1, endpoint=True)

    # Calculate the metrics
    precision, recall, scores, iou_lookup = evaluate_metrics(cocoEval, params, args.show_eval_summary)

    # take precision for all classes, all areas and 100 detections
    display_metrics(precision, recall, scores, iou_lookup, log_path=args.output_log_path)

    # Calculate metrics for each category
    for cat in coco_gt.loadCats(coco_gt.getCatIds()):
        # Calculate the metrics
        params.catIds = [cat["id"]]
        precision, recall, scores, iou_lookup = evaluate_metrics(cocoEval, params, args.show_eval_summary)
        # take precision for all classes, all areas and 100 detections
        display_metrics(precision, recall, scores, iou_lookup, class_name=cat["name"], log_path=args.output_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Metrics from the predictions and Ground Truths")
    parser.add_argument("--gt-coco-path", default="D:/mmdetection/data/OPT/annotations/instances_val.json", type=str)
    parser.add_argument("--evaluation-result-path", default="D:/YOLO-MIF-master/runs/val/YOLOv8-allfusion/predictions1.json", type=str)
    parser.add_argument("--output-log-path", type=str, default="evaluation.log")
    parser.add_argument("--show-eval-summary", type=bool, default=True)
    parser.add_argument("--gt-eval-number", type=int, default=1411)

    args = parser.parse_args()
    input_json = args.evaluation_result_path  # 输入JSON文件路径

    det_bbox = set_category_id(input_json, input_json, target_id=1)
    _compute_and_display_metrics(args)