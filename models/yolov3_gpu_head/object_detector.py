from models.yolov3_gpu_head.inference import infer_objects_in_image, restore_model
from models.yolov3_gpu_head.conversion.utils import load_classes
from utils.image import load_pil_image_from_file
from utils.preprocess_image import resize_and_letter_box
import numpy as np
import keras.backend as K

NUM_OF_CLASSES = 601
NUM_OF_ANCHORS = 3
DEFAULT_MODEL_IMAGE_WIDTH = 608
DEFAULT_MODEL_IMAGE_HEIGHT = 608
OPENIMAGES_ANCHORS = np.array([
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]]
])


class ObjectDetector:

    name = 'YOLOv3'

    def __init__(
        self,
        *,
        detection_threshold: float =0.5,
        nms_threshold: float =0.6,
        anchors=OPENIMAGES_ANCHORS,
        model_image_width: int = DEFAULT_MODEL_IMAGE_WIDTH,
        model_image_height: int = DEFAULT_MODEL_IMAGE_HEIGHT,
        path_to_model: str = None,
        path_to_classes: str = None
    ):
        self.model = restore_model() if path_to_model is None else restore_model(path_to_model)
        self.class_index_to_human_readable_class = load_classes() if path_to_classes is None else load_classes(path_to_classes)

        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.model_image_width = model_image_width
        self.model_image_height = model_image_height

        self.session = K.get_session()
        self.anchors = anchors / np.array([model_image_width, model_image_height])
    #     self.outputs = self.generate_session_outputs()
    #
    # def generate_session_outputs(self):
    #     boxes_xy = []
    #     boxes_wh = []
    #     classes_probs = []
    #     reshaped_heads = []
    #
    #     restored_model = self.model
    #
    #     for yolo_head_idx in range(len(restored_model.output)):
    #         yolo_head = restored_model.output[yolo_head_idx]
    #         yolo_head_shape = K.shape(yolo_head)
    #         yolo_head_num_of_cols, yolo_head_num_of_rows = yolo_head_shape[1], yolo_head_shape[2]
    #
    #         curr_yolo_head = K.reshape(yolo_head, [-1, yolo_head_num_of_cols, yolo_head_num_of_rows, NUM_OF_ANCHORS,
    #                                                5 + NUM_OF_CLASSES])
    #         reshaped_heads.append(curr_yolo_head)
    #
    #         grid = K.cast(get_grid(yolo_head_shape[1], yolo_head_shape[2]), dtype=K.dtype(curr_yolo_head))
    #
    #         curr_boxes_xy = (K.sigmoid(curr_yolo_head[..., :2]) + grid) / K.cast(
    #             [yolo_head_shape[1], yolo_head_shape[2]], dtype=K.dtype(curr_yolo_head))
    #         curr_boxes_wh = K.exp(curr_yolo_head[..., 2:4]) * ANCHORS[yolo_head_idx]
    #         curr_prob_obj = K.sigmoid(curr_yolo_head[..., 4])
    #         curr_prob_class = K.sigmoid(curr_yolo_head[..., 5:])
    #
    #         curr_prob_detected_class = K.tile(
    #             K.reshape(curr_prob_obj, [-1, K.shape(curr_prob_obj)[1], K.shape(curr_prob_obj)[2], NUM_OF_ANCHORS, 1]),
    #             [1, 1, 1, 1, NUM_OF_CLASSES]) * curr_prob_class
    #
    #         boxes_xy.append(curr_boxes_xy)
    #         boxes_wh.append(curr_boxes_wh)
    #         classes_probs.append(curr_prob_detected_class)
    #
    #     return boxes_xy, boxes_wh, classes_probs

    def infer_object_detections_on_loaded_image(
        self,
        image_np: np.array,
    ):
        orig_img_height, orig_img_width = image_np.shape[:2]
        img_np = resize_and_letter_box(image_np/255., target_width=self.model_image_width, target_height=self.model_image_height)
        img_np = np.expand_dims(img_np, 0)

        detected_boxes, detected_classes, detected_scores = infer_objects_in_image(
            image=img_np,
            restored_model=self.model,
            session=self.session,
            orig_image_height=orig_img_height,
            orig_image_width=orig_img_width,
            detection_prob_treshold=self.detection_threshold,
            model_image_height=self.model_image_height,
            model_image_width=self.model_image_width,
            anchors=self.anchors
        )

        human_readable_classes = []
        for detected_class_for_img in detected_classes:
            human_readable_classes.append(self.class_index_to_human_readable_class[detected_class_for_img])

        return detected_boxes, human_readable_classes, detected_scores

    def infer_object_detections(self, target_file_path: str):
        _, img_np = load_pil_image_from_file(target_file_path)
        return self.infer_object_detections_on_loaded_image(np.array(img_np))


