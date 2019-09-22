from models.yolov3_gpu_head.inference import infer_objects_in_image, restore_model
from models.yolov3_gpu_head.conversion.utils import load_classes
from models.preprocessing.letterbox import resize_and_letter_box
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from models.data.base_object_detector import BaseObjectDetector


NUM_OF_CLASSES = 601
NUM_OF_ANCHORS = 3
DEFAULT_MODEL_IMAGE_WIDTH = 608
DEFAULT_MODEL_IMAGE_HEIGHT = 608
OPENIMAGES_ANCHORS = np.array([
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [59, 119]],
    [[10, 13], [16, 30], [33, 23]]
])


class ObjectDetector(BaseObjectDetector):

    name = 'YOLOv3'

    def __init__(
        self,
        *,
        detection_threshold: float = 0.5,
        nms_threshold: float = 0.6,
        anchors=OPENIMAGES_ANCHORS,
        model_image_width: int = DEFAULT_MODEL_IMAGE_WIDTH,
        model_image_height: int = DEFAULT_MODEL_IMAGE_HEIGHT,
        path_to_model: str = None,
        path_to_classes: str = None,
        log_device_placement: bool = True,
        gpu_allow_growth: bool = True
    ):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = gpu_allow_growth  # dynamically grow the memory used on the GPU
        config.log_device_placement = log_device_placement  # to log device placement (on which device the operation ran)
        # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        self.session = K.get_session()

        self.model = restore_model() if path_to_model is None else restore_model(path_to_model)
        self.class_index_to_human_readable_class = load_classes() if path_to_classes is None else load_classes(path_to_classes)

        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.model_image_width = model_image_width
        self.model_image_height = model_image_height

        self.anchors = anchors / np.array([model_image_width, model_image_height])

    def infer_object_detections_on_loaded_image(
        self,
        image_np: np.array,
    ):
        orig_img_height, orig_img_width = image_np.shape[:2]
        img_np = resize_and_letter_box(image_np / 255., target_width=self.model_image_width, target_height=self.model_image_height)
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

        return detected_boxes, detected_classes, detected_scores

    def get_mapping_from_class_idx_to_readable_class(self):
        return self.class_index_to_human_readable_class
