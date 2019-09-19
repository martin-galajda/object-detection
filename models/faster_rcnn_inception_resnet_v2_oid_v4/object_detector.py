from models.faster_rcnn_inception_resnet_v2_oid_v4.inference \
    import _construct_session_for_inference, infer_objects_in_image
from models.faster_rcnn_inception_resnet_v2_oid_v4.utils import build_class_index
import numpy as np
from models.data.base_object_detector import BaseObjectDetector
from models.faster_rcnn_inception_resnet_v2_oid_v4.constants import FasterRCNNPathConstants


class ObjectDetector(BaseObjectDetector):
    name = 'FasterRCNN'

    def __init__(
        self,
        *,
        path_to_frozen_graph: str = FasterRCNNPathConstants.PATH_TO_FROZEN_TF_GRAPH,
        path_to_classes: str = None,
        use_gpu: bool = True,
        log_device_placement: bool = True
    ):
        self.session = _construct_session_for_inference(
            path_to_frozen_inference_graph=path_to_frozen_graph,
            use_gpu=use_gpu,
            log_device=log_device_placement
        )

        self.class_index_to_human_readable_class = build_class_index() \
            if path_to_classes is None \
            else build_class_index(path_to_classes)

    def close(self):
        self.session.close()

    def infer_object_detections_on_loaded_image(self, img_np: np.array):
        orig_img_height, orig_img_width = img_np.shape[:2]
        img_np = np.expand_dims(img_np, 0)
        output_dict = infer_objects_in_image(
            image=img_np,
            sess=self.session
        )

        detected_classes = []
        detected_boxes = []
        detected_scores = []
        num_detections = output_dict['num_detections']

        for i in range(num_detections):
            class_idx = output_dict['detection_classes'][i]
            detected_classes.append(class_idx)
            detected_scores.append(output_dict['detection_scores'][i])

            detected_box = output_dict['detection_boxes'][i]
            top, left, bottom, right = detected_box

            img_width, img_height = orig_img_width, orig_img_height

            detected_boxes.append([
                left * img_width,   # x1
                top * img_height,   # y1
                right * img_width,  # x2
                bottom * img_height  # y2
            ])

        return detected_boxes, detected_classes, detected_scores

    def get_mapping_from_class_idx_to_readable_class(self):
        return self.class_index_to_human_readable_class
