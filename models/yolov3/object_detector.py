from models.yolov3.inference import infer_objects_in_image, restore_model
from models.yolov3.conversion.utils import load_classes
from utils.image import load_pil_image_from_file
from utils.preprocess_image import resize_and_letter_box
import numpy as np


class ObjectDetector:

    def __init__(self, *, detection_prob_treshold=0.25):
        self.model = restore_model()
        self.class_index_to_human_readable_class = load_classes()
        self.detection_prob_treshold = detection_prob_treshold

    def infer_object_detections(self, target_file_path: str):
        orig_img_pil, _ = load_pil_image_from_file(target_file_path)
        img_np = resize_and_letter_box(np.array(orig_img_pil)/255., target_width = 608, target_height = 608)
        img_np = np.expand_dims(img_np, 0)
        detected_boxes, detected_classes, detected_scores = infer_objects_in_image(
            image=img_np*255.,
            model=self.model,
            orig_image_height=orig_img_pil.height,
            orig_image_width=orig_img_pil.width,
            detection_prob_treshold = self.detection_prob_treshold)

        human_readable_classes = []
        for detected_classes_for_img in detected_classes:
            curr_classes = []
            for detected_class_for_img in detected_classes_for_img:
                curr_classes.append(self.class_index_to_human_readable_class[detected_class_for_img])
            human_readable_classes.append(curr_classes)

        return detected_boxes, human_readable_classes, detected_scores

