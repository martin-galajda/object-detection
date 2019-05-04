import numpy as np
from utils.compute_iou import compute_iou

def compute_basic_precision(
    predicted_boxes: np.ndarray,
    ground_truth_boxes: np.ndarray,
    min_iou_for_detection: float = 0.5
):
    """
    Computes basic precision (num_true_positives / num_ground_truth_positives).

    Parameters:
        - predicted_boxes: numpy array of shape [batch_size, max_boxes_detected, 5] where:
            - 2nd dimension contains maximum number of boxes detected for some sample in batch
              samples from batch having less detected boxes are zero-padded
            - 3rd dimension with 5 elements consists of - [x_min, y_min, x_max, y_max, class]
            - all coordinates are expected to be from [0, 1] interval
            - class is expected to be integer from interval [1, number_of_classes]

        - ground_truth_boxes: numpy array of shape [batch_size, max_ground_truth_boxes_for_img, 5]
            - 2nd dimension contains maximum number of ground truth boxes for some sample img in batch, zero-padded
            - 3rd dimension with 5 elements consists of - [x_min, y_min, x_max, y_max, class]
            - class is expected to be integer from interval [1, number_of_classes]

        - min_iou_for_detection:
            Minimum IOU (intersection over union) between predicted and
            ground truth box for considering box as detected
    Returns:
        - basic_precision: float: true_positives / ground_truth_boxes
    """

    zero_padded_box = np.zeros(5)
    batch_size = ground_truth_boxes.shape[0]
    max_ground_truth_boxes = ground_truth_boxes.shape[1]

    real_ground_truth_boxes_mask = np.zeros((batch_size, max_ground_truth_boxes))
    detected_ground_truth_boxes_count = np.zeros((batch_size, max_ground_truth_boxes))

    ground_truth_box_already_detected = np.zeros((batch_size, max_ground_truth_boxes))

    # detected boxes true positives
    detected_boxes_tp = np.zeros((int(batch_size), int(max_ground_truth_boxes)))

    max_predicted_boxes = predicted_boxes.shape[1]

    # detected boxes false positives
    detected_boxes_fp = np.zeros((batch_size, max_predicted_boxes))

    overall_basic_accuracy = 0.0

    for image_idx, _ in enumerate(predicted_boxes):
        predicted_image_boxes = predicted_boxes[image_idx]
        image_ground_truth_boxes = ground_truth_boxes[image_idx]

        for predicted_box_idx, predicted_image_box in enumerate(predicted_image_boxes):
            if np.array_equal(predicted_image_box, zero_padded_box):
                continue

            predicted_box_ious = []
            for ground_truth_box_idx, image_ground_truth_box in enumerate(image_ground_truth_boxes):
                if np.array_equal(image_ground_truth_box, zero_padded_box):
                    predicted_box_ious += [0.0]
                    continue
                real_ground_truth_boxes_mask[image_idx, ground_truth_box_idx] = 1.0
                predicted_box_ious += [compute_iou(image_ground_truth_box[1:], predicted_image_box[1:])]

            max_iou_ground_truth_box_idx = np.argmax(predicted_box_ious)
            max_iou = predicted_box_ious[max_iou_ground_truth_box_idx]
            ground_truth_box = image_ground_truth_boxes[max_iou_ground_truth_box_idx]

            classes_match = int(ground_truth_box[0]) == int(predicted_image_box[0])

            if ground_truth_box_already_detected[image_idx, max_iou_ground_truth_box_idx] == 1.0 or not classes_match:
                detected_boxes_fp[image_idx, predicted_box_idx] = 1.0
            else:
                detected_boxes_tp[image_idx, predicted_box_idx] = 1.0
                ground_truth_box_already_detected[image_idx, max_iou_ground_truth_box_idx] = 1.0

        matched_count = len(np.where(detected_boxes_tp[image_idx, :] == 1.0)[0])
        ground_truth_count = len(np.where(real_ground_truth_boxes_mask[image_idx, :] == 1.0)[0])
        sample_basic_accuracy = matched_count / ground_truth_count

        print(f'sample_basic_accuracy = {sample_basic_accuracy}')
        overall_basic_accuracy += sample_basic_accuracy

    overall_basic_accuracy /= batch_size

    return overall_basic_accuracy