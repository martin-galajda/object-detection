from models.yolov3.constants import PathConstants
from tensorflow.keras.models import load_model
from utils.non_max_suppression import classic_non_max_suppression
import time
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from typing import Tuple

NUM_OF_CLASSES = 601
NUM_OF_BOX_PARAMS = 5
NUM_OF_ANCHORS = 3


def restore_model(path_to_model = PathConstants.YOLOV3_MODEL_OPENIMAGES_OUT_PATH):
    return load_model(path_to_model)

def construct_grid(rows, cols):
    grid_x = K.arange(0, stop=cols)
    grid_x = K.reshape(grid_x, [1, -1, 1, 1])
    grid_x = K.tile(grid_x, [rows, 1, 1, 1])

    grid_y = K.arange(0, stop=rows)
    grid_y = K.reshape(grid_y, [-1, 1, 1, 1])
    grid_y = K.tile(grid_y, [1, cols, 1, 1])

    grid = K.concatenate([grid_x, grid_y])

    return grid


def _infer_network_outputs(
    *,
    sess,
    restored_model,
    num_of_anchors,
    anchors,
    orig_image_width,
    orig_image_height,
    model_image_width,
    model_image_height,
    img_np
):
    start = time.time()
    boxes = []
    prob_class = []

    for yolo_head_idx in range(len(restored_model.output)):
        yolo_head = restored_model.output[yolo_head_idx]
        yolo_head_shape = K.shape(yolo_head)
        yolo_head_num_of_cols, yolo_head_num_of_rows = yolo_head_shape[2], yolo_head_shape[1]

        curr_yolo_head = K.reshape(yolo_head, [-1, yolo_head_num_of_cols, yolo_head_num_of_rows, num_of_anchors,
                                               NUM_OF_BOX_PARAMS + NUM_OF_CLASSES])

        grid = construct_grid(yolo_head_shape[1], yolo_head_shape[2])
        grid = K.cast(grid, dtype=K.dtype(curr_yolo_head))
        grid_size = K.cast([yolo_head_num_of_cols, yolo_head_num_of_rows], dtype=K.dtype(curr_yolo_head))

        curr_boxes_xy = (K.sigmoid(curr_yolo_head[..., :2]) + grid) / grid_size

        curr_boxes_wh = K.exp(curr_yolo_head[..., 2:4]) * anchors[yolo_head_idx]

        curr_prob_obj = K.sigmoid(curr_yolo_head[..., 4:5])
        curr_prob_class = K.sigmoid(curr_yolo_head[..., 5:])
        curr_prob_detected_class = curr_prob_obj * curr_prob_class

        boxes.append(get_corrected_boxes(
            box_width=curr_boxes_wh[..., 0:1],
            box_height=curr_boxes_wh[..., 1:2],
            box_x=curr_boxes_xy[..., 0:1],
            box_y=curr_boxes_xy[..., 1:2],
            orig_image_shape=(orig_image_width, orig_image_height),
            model_image_shape=(model_image_width, model_image_height)
        ))

        curr_prob_detected_class = K.reshape(curr_prob_detected_class, [-1, NUM_OF_CLASSES])
        prob_class.append(curr_prob_detected_class)

    prob_class = K.concatenate(prob_class, axis=0)
    boxes = K.concatenate(boxes, axis=0)

    out_tensors = [
        boxes,
        prob_class,
    ]

    print(f'Took {time.time() - start} seconds to construct network.')

    start = time.time()
    sess_out = sess.run(out_tensors, feed_dict={
        restored_model.input: img_np,
        K.learning_phase(): 0
    })

    print(f'Took {time.time() - start} seconds to infer outputs in session.')
    boxes, out_boxes_classes = sess_out
    return boxes, out_boxes_classes


def infer_objects_in_image(
    *,
    image: np.array,
    session: tf.Session,
    orig_image_height: int,
    orig_image_width: int,
    detection_prob_treshold=0.5,
    nms_threshold=0.6,
    model_image_height: int,
    model_image_width: int,
    anchors: np.array,
    restored_model: tf.keras.Model,
    num_of_anchors: int = NUM_OF_ANCHORS,
    num_of_classes=NUM_OF_CLASSES
):
    boxes, classes_probs = _infer_network_outputs(
        sess=session,
        model_image_height=model_image_height,
        model_image_width=model_image_width,
        anchors=anchors,
        img_np=image,
        orig_image_width=orig_image_width,
        orig_image_height=orig_image_height,
        restored_model=restored_model,
        num_of_anchors=num_of_anchors
    )

    all_curr_detected_objects = []
    all_curr_detected_classes = []
    all_curr_detected_scores = []

    for c in range(num_of_classes):
        curr_mask_detected = classes_probs[..., c] > detection_prob_treshold
        curr_probs_class = classes_probs[curr_mask_detected, :][:, c]
        c_boxes = boxes[curr_mask_detected, :]

        curr_detected_objects = []
        curr_detected_classes = []
        curr_detected_scores = []

        for idx in range(np.count_nonzero(curr_mask_detected)):
            box_class_prob = curr_probs_class[idx]

            curr_detected_objects += [c_boxes[idx]]
            curr_detected_classes += [c]
            curr_detected_scores += [box_class_prob]

        if len(curr_detected_objects) > 0:
            chosen_box_indices = classic_non_max_suppression(curr_detected_objects, curr_detected_scores, nms_threshold)
            curr_detected_objects = [curr_detected_objects[i] for i in chosen_box_indices]
            curr_detected_classes = [curr_detected_classes[i] for i in chosen_box_indices]
            curr_detected_scores = [curr_detected_scores[i] for i in chosen_box_indices]

            all_curr_detected_objects += curr_detected_objects
            all_curr_detected_classes += curr_detected_classes
            all_curr_detected_scores += curr_detected_scores

    return all_curr_detected_objects, all_curr_detected_classes, all_curr_detected_scores


def get_corrected_boxes(
    *,
    box_width: tf.Tensor,
    box_height: tf.Tensor,
    box_x: tf.Tensor,
    box_y: tf.Tensor,
    orig_image_shape: Tuple[tf.Tensor],
    model_image_shape: Tuple[float]
):
    """
    Post-process outputs produced by YOLOv3 CNN network.
    We letter-box and resize image into fixed size.
    The function transforms predictions into original dimensions of the image.

    :param box_width: predicted box widths by YOLOv3
    :param box_height: predicted box heights by YOLOv3
    :param box_x: predicted x coordinates of the center of the box
    :param box_y: predicted y coordinates of the center of the box
    :param orig_image_shape: (width, height) of original image
    :param model_image_shape: (width, height) of resized image used as input to the model
    :return: corrected boxes to match original dimensions of image
    """
    orig_image_w, orig_image_h = orig_image_shape[0], orig_image_shape[1]
    model_w, model_h = model_image_shape[0], model_image_shape[1]

    if float(model_w / orig_image_w) < float(model_h / orig_image_h):
        w_without_padding = model_w
        h_without_padding = (orig_image_h) * model_w / orig_image_w
    else:
        h_without_padding = model_h
        w_without_padding = (orig_image_w) * model_h / orig_image_h

    x_shift = (model_w - w_without_padding) / 2.0 / model_w
    y_shift = (model_h - h_without_padding) / 2.0 / model_h

    box_x = (box_x - x_shift) / (w_without_padding / model_w)
    box_y = (box_y - y_shift) / (h_without_padding / model_h)

    box_width *= model_w / w_without_padding
    box_height *= model_h / h_without_padding

    left = (box_x - (box_width / 2.)) * orig_image_w
    right = (box_x + (box_width / 2.)) * orig_image_w
    top = (box_y - (box_height / 2.)) * orig_image_h
    bottom = (box_y + (box_height / 2.)) * orig_image_h

    output_boxes = K.concatenate([
        K.reshape(left, [-1, 1]),
        K.reshape(top, [-1, 1]),
        K.reshape(right, [-1, 1]),
        K.reshape(bottom, [-1, 1])
    ])

    return output_boxes

