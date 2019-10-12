from models.data.bounding_box import BoundingBox
from typing import List
from collections import defaultdict
from utils.compute_iou import compute_iou
import numpy as np
import math
from evaluation.average_precision.results import EvaluationResults


def calculateInterpolatedAveragePrecision(
    recalls,
    precisions
):
    """
    Calculates interpolated average precision according to PASCAL VOC competition.

    :param recalls     list of all recalls, sorted ascendingly
    :param precisions  list of all precisions, with order matching corresponding recall values

    """
    recall_values = [0]
    recall_values += list(recalls)
    recall_values.append(1)

    interp_precision_values = [0.]
    interp_precision_values += list(precisions)
    interp_precision_values.append(0.)

    for i in range(len(interp_precision_values) - 1, 0, -1):
        interp_precision_values[i - 1] = max(interp_precision_values[i - 1], interp_precision_values[i])

    unique_recall_value_indices = []

    for recall_value_idx in range(len(recall_values) - 1):
        if recall_values[recall_value_idx + 1] != recall_values[recall_value_idx]:
            unique_recall_value_indices.append(recall_value_idx + 1)

    AP = 0
    for unique_recall_value_index in unique_recall_value_indices:
        recall_length = recall_values[unique_recall_value_index] - recall_values[unique_recall_value_index - 1]
        AP = AP + (recall_length * interp_precision_values[unique_recall_value_index])

    return (AP, recall_values, interp_precision_values)


class Evaluator:
    """
    Computes average precision based on
    provided list of detected and ground truth boxes.
    """
    detected_bboxes: List[BoundingBox]
    ground_truth_bboxes: List[BoundingBox]

    def __init__(
        self, 
        *,
        detected_bounding_boxes: List[BoundingBox],
        ground_truth_bounding_boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
        drop_empty_gt_files: bool = True
    ):
        self.detected_bboxes = detected_bounding_boxes
        self.ground_truth_bboxes = ground_truth_bounding_boxes
        self.iou_threshold = iou_threshold
        self.drop_empty_gt_files = drop_empty_gt_files

    def evaluate(self) -> EvaluationResults:
        """
        Computes AP, interpolated recalls and precisions,
            number of true positive (TP) and number of false positives (FP) for each class 
            based on provided boxes and IOU threshold
            and returns it as EvaluationResults object.
        """
        detected_bboxes_by_class = defaultdict(list)
        gt_bboxes_by_class = defaultdict(list)
        gt_bboxes_by_filename = defaultdict(list)

        all_classes_indices = set()

        for detected_bbox in self.detected_bboxes:
            detected_bboxes_by_class[detected_bbox.human_readable_class].append(detected_bbox)
            all_classes_indices.add(detected_bbox.human_readable_class)

        for gt_bbox in self.ground_truth_bboxes:
            gt_bboxes_by_class[gt_bbox.human_readable_class].append(gt_bbox)
            gt_bboxes_by_filename[gt_bbox.filename].append(gt_bbox)
            all_classes_indices.add(gt_bbox.human_readable_class)

        AP_per_class = defaultdict(lambda: 0.)
        interpolated_recalls_per_class = {}
        interpolated_precisions_per_class = {}
        TP_per_class = {}
        FP_per_class = {}

        for class_idx in all_classes_indices:
            current_detected_bboxes = detected_bboxes_by_class[class_idx]

            def has_gt_object_in_file(bbox: BoundingBox):
                return len(gt_bboxes_by_filename[bbox.filename]) > 0

            if self.drop_empty_gt_files:
                current_detected_bboxes = list(filter(has_gt_object_in_file, current_detected_bboxes))
            
            current_gt_bboxes = gt_bboxes_by_class[class_idx]

            if len(current_gt_bboxes) == 0:
                continue

            current_gt_bboxes_by_filename = defaultdict(list)

            used_gt_boxes = defaultdict(lambda: defaultdict(lambda: False))

            TP = np.zeros(len(current_detected_bboxes))
            FP = np.zeros(len(current_detected_bboxes))

            for current_gt_bbox in current_gt_bboxes:
                current_gt_bboxes_by_filename[current_gt_bbox.filename].append(current_gt_bbox)

            current_detected_bboxes = list(sorted(current_detected_bboxes, key = lambda bbox: bbox.score, reverse=True))

            for curr_detected_idx, current_detected_bbox in enumerate(current_detected_bboxes):
                curr_image_gt_bboxes = current_gt_bboxes_by_filename[current_detected_bbox.filename]

                max_iou = -math.inf
                max_iou_gt_box_idx = -1

                for idx, curr_image_gt_bbox in enumerate(curr_image_gt_bboxes):
                    iou = compute_iou(
                        [
                            current_detected_bbox.min_x,
                            current_detected_bbox.min_y,
                            current_detected_bbox.max_x,
                            current_detected_bbox.max_y
                        ],
                        [
                            curr_image_gt_bbox.min_x,
                            curr_image_gt_bbox.min_y,
                            curr_image_gt_bbox.max_x,
                            curr_image_gt_bbox.max_y
                        ],
                    )

                    if iou > max_iou:
                        max_iou = iou
                        max_iou_gt_box_idx = idx

                if max_iou < self.iou_threshold or used_gt_boxes[current_detected_bbox.filename][max_iou_gt_box_idx] is True:
                    FP[curr_detected_idx] = 1
                else:
                    TP[curr_detected_idx] = 1
                    used_gt_boxes[current_detected_bbox.filename][max_iou_gt_box_idx] = True

            cumulated_TP = np.cumsum(TP)
            cumulated_FP = np.cumsum(FP)

            recalls = cumulated_TP / len(current_gt_bboxes)
            precisions = cumulated_TP / (cumulated_TP + cumulated_FP)

            AP, recall_values, interp_precision_vals = calculateInterpolatedAveragePrecision(
                recalls=recalls,
                precisions=precisions
            )

            AP_per_class[class_idx] = AP
            interpolated_recalls_per_class[class_idx] = recall_values
            interpolated_precisions_per_class[class_idx] = interp_precision_vals
            TP_per_class[class_idx] = int(np.sum(TP))
            FP_per_class[class_idx] = int(np.sum(FP))

        eval_results = EvaluationResults(
            AP_per_class=AP_per_class,
            recalls_per_class=interpolated_recalls_per_class,
            interp_precisions_per_class=interpolated_precisions_per_class,
            TP_per_class=TP_per_class,
            FP_per_class=FP_per_class
        )

        return eval_results
