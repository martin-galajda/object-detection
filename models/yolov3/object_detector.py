from models.yolov3.inference import infer_objects_in_image, restore_model
from models.yolov3.conversion.utils import load_classes
from utils.image import load_pil_image_from_file
from utils.preprocess_image import resize_and_letter_box
import numpy as np


class ObjectDetector:

    name = 'YOLOv3'

    def __init__(self, *,
                 detection_prob_treshold=0.5,
                 path_to_model: str = None,
                 path_to_classes: str = None):
        self.model = restore_model() if path_to_model is None else restore_model(path_to_model)
        self.class_index_to_human_readable_class = load_classes() if path_to_classes is None else load_classes(path_to_classes)
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

        human_readable_classes = []
        for detected_classes_for_img in detected_classes:
            curr_classes = []
            for detected_class_for_img in detected_classes_for_img:
                curr_classes.append(self.class_index_to_human_readable_class[detected_class_for_img])
            human_readable_classes.append(curr_classes)

        return detected_boxes, human_readable_classes, detected_scores

    def infer_object_detections(self, target_file_path: str):
        orig_img_pil, _ = load_pil_image_from_file(target_file_path)
        return self.infer_object_detections_on_loaded_image(np.array(orig_img_pil))
