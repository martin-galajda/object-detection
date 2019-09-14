from models.yolov3_gpu_head_v2.constants import PathConstants
from keras.models import load_model
from utils.non_max_suppression import classic_non_max_suppression
import time
import keras.backend as K
import tensorflow as tf
import numpy as np


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


def _construct_out_tensors(
    *,
    restored_model,
    num_of_anchors,
    anchors,
    model_image_width,
    model_image_height,
    prob_detection_threshold = 0.25,
    nms_iou_threshold = 0.5
):
    start = time.time()
    boxes = []
    prob_class = []

    placeholder_orig_image_width = K.placeholder(shape=(1,))
    placeholder_orig_image_height = K.placeholder(shape=(1,))

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
            orig_image_shape=(placeholder_orig_image_width, placeholder_orig_image_height),
            model_image_shape=(model_image_width, model_image_height)
        ))

        curr_prob_detected_class = K.reshape(curr_prob_detected_class, [-1, NUM_OF_CLASSES])
        prob_class.append(curr_prob_detected_class)

    prob_class = K.concatenate(prob_class, axis=0)
    boxes = K.concatenate(boxes, axis=0)

    mask = prob_class >= prob_detection_threshold
    max_boxes_tensor = K.constant(20, dtype='int32')

    picked_boxes = []
    picked_scores = []
    picked_classes = []

    for c in range(NUM_OF_CLASSES):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(prob_class[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                 iou_threshold=nms_iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        picked_boxes.append(class_boxes)
        picked_scores.append(class_box_scores)
        picked_classes.append(classes)

    picked_boxes = K.concatenate(picked_boxes, axis=0)
    picked_scores = K.concatenate(picked_scores, axis=0)
    picked_classes = K.concatenate(picked_classes, axis=0)

    out_tensors = [
        picked_boxes,
        picked_scores,
        picked_classes
    ]

    print(f'Took {time.time() - start} seconds to construct network.')

    input_tensors = [
        restored_model.input,
        placeholder_orig_image_width,
        placeholder_orig_image_height
    ]

    return out_tensors, input_tensors


def infer_objects_in_image(
    *,
    image: np.array,
    session,
    out_tensors,
    input_tensor,
    orig_image_height_placeholder_tensor,
    orig_image_width_placeholder_tensor,
    orig_image_height: int,
    orig_image_width: int,
):
    boxes, scores, classes = session.run(out_tensors, feed_dict={
        orig_image_height_placeholder_tensor: [orig_image_height],
        orig_image_width_placeholder_tensor: [orig_image_width],
        input_tensor: image,
        K.learning_phase(): 0
    })

    return boxes, classes, scores


def get_corrected_boxes(*, box_width, box_height, box_x, box_y, orig_image_shape, model_image_shape):
    orig_image_w, orig_image_h = orig_image_shape[0], orig_image_shape[1]
    model_w, model_h = model_image_shape[0], model_image_shape[1]

    scale = K.min(
        K.concatenate([(model_w / orig_image_w), (model_h / orig_image_h)])
    )
    w_without_padding = orig_image_w * scale
    h_without_padding = orig_image_h * scale

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

