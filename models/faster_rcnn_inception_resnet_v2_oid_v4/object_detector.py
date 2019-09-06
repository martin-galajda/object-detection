from models.faster_rcnn_inception_resnet_v2_oid_v4.inference import infer_objects_in_image, restore_inference_graph
from models.faster_rcnn_inception_resnet_v2_oid_v4.utils import build_class_index
from utils.image import load_pil_image_from_file
import numpy as np


class ObjectDetector:
    name = 'FasterRCNN'

    def __init__(self):
        self.inference_graph = restore_inference_graph()
        self.class_index_to_human_readable_class = build_class_index()

    def infer_object_detections(self, target_file_path: str):
        orig_img_pil, img_np = load_pil_image_from_file(target_file_path)
        img_np = np.expand_dims(img_np, 0)
        output_dict = infer_objects_in_image(image=img_np, inference_graph=self.inference_graph)

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

            img_width, img_height = orig_img_pil.width, orig_img_pil.height

            detected_boxes.append([
                left * img_width,  # x1
                top * img_height,  # y1
                right * img_width,  # x2
                bottom * img_height  # y2
            ])

        detected_boxes = np.array(detected_boxes)

        return detected_boxes, detected_classes, detected_scores

