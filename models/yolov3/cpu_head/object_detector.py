from models.yolov3.cpu_head.inference import infer_objects_in_image, restore_model
from models.yolov3.conversion.utils import load_classes
from models.preprocessing.letterbox import resize_and_letter_box
from models.data.base_object_detector import BaseObjectDetector
import numpy as np


class ObjectDetector(BaseObjectDetector):

    name = 'YOLOv3'

    def __init__(
        self,
        *,
        detection_prob_treshold=0.5,
        path_to_model: str = None,
        path_to_classes: str = None
    ):
        self.model = restore_model() if path_to_model is None else restore_model(path_to_model)
        self.class_index_to_human_readable_class = load_classes() \
            if path_to_classes is None \
            else load_classes(path_to_classes)
        self.detection_prob_treshold = detection_prob_treshold

    def infer_object_detections_on_loaded_image(
        self,
        image_np: np.array,
    ):
        orig_img_height, orig_img_width = image_np.shape[:2]
        img_np = resize_and_letter_box(image_np / 255., target_width = 608, target_height = 608)
        img_np = np.expand_dims(img_np, 0)
        detected_boxes, detected_classes, detected_scores = infer_objects_in_image(
            image =img_np * 255.,
            model=self.model,
            orig_image_height=orig_img_height,
            orig_image_width=orig_img_width,
            detection_prob_treshold = self.detection_prob_treshold)

        return detected_boxes, detected_classes, detected_scores

    def get_mapping_from_class_idx_to_readable_class(self):
        return self.class_index_to_human_readable_class
