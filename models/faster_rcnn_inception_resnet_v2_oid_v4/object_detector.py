from models.faster_rcnn_inception_resnet_v2_oid_v4.inference import infer_objects_in_image, restore_inference_graph
from models.faster_rcnn_inception_resnet_v2_oid_v4.utils import build_class_index
from utils.image import load_pil_image_from_file
import numpy as np


class ObjectDetector:
    name = 'FasterRCNN'

    def __init__(
        self,
        path_to_frozen_graph: str = None,
        path_to_classes: str = None,
        session=None,
        use_gpu: bool = False
    ):
        self.inference_graph = restore_inference_graph() \
            if path_to_frozen_graph is None \
            else restore_inference_graph(path_to_frozen_graph, use_gpu)
        self.class_index_to_human_readable_class = build_class_index() \
            if path_to_classes is None \
            else build_class_index(path_to_classes)
        self.session = session

    def infer_object_detections_on_loaded_image(self, img_np: np.array):
        orig_img_width, orig_img_height = img_np.shape[:2]
        img_np = np.expand_dims(img_np, 0)
        output_dict = infer_objects_in_image(image=img_np,
                                             inference_graph=self.inference_graph,
                                             session=self.session)

        detected_classes = []
        detected_boxes = []
        detected_scores = []
        num_detections = output_dict['num_detections']

        for i in range(num_detections):
            class_idx = output_dict['detection_classes'][i]
            detected_classes.append([self.class_index_to_human_readable_class[class_idx]])
            detected_scores.append([output_dict['detection_scores'][i]])

            detected_box = output_dict['detection_boxes'][i]
            top, left, bottom, right = detected_box

            img_width, img_height = orig_img_width, orig_img_height

            detected_boxes.append([
                left * img_width,   # x1
                top * img_height,   # y1
                right * img_width,  # x2
                bottom * img_height # y2
            ])

        detected_boxes = np.array(detected_boxes)

        return detected_boxes, detected_classes, detected_scores

    def infer_object_detections(self, target_file_path: str):
        orig_img_pil, img_np = load_pil_image_from_file(target_file_path)
        return self.infer_object_detections_on_loaded_image(img_np)

