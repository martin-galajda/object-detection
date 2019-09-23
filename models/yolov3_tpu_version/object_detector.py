from models.yolov3_tpu_version.inference import infer_objects_in_image, restore_model, _construct_inference_tensors
from models.yolov3_tpu_version.conversion.utils import load_classes
from models.preprocessing.letterbox import resize_and_letter_box
from models.data.base_object_detector import BaseObjectDetector
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.backend.tensorflow_backend import set_session
import time


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
        detection_threshold: float = 0.3,
        nms_threshold: float = 0.6,
        anchors=OPENIMAGES_ANCHORS,
        model_image_width: int = DEFAULT_MODEL_IMAGE_WIDTH,
        model_image_height: int = DEFAULT_MODEL_IMAGE_HEIGHT,
        path_to_model: str = None,
        path_to_classes: str = None,
        log_device_placement: bool = True,
        gpu_allow_growth: bool = True,
        verbose: bool = True,
    ):

        config = tf.ConfigProto()

        # dynamically grow the memory used on the GPU
        config.gpu_options.allow_growth = gpu_allow_growth

        # log device placement (on which device the operation ran)
        config.log_device_placement = log_device_placement

        # create and set this TensorFlow session as the default session for Keras
        sess = tf.Session(config=config)
        set_session(sess)

        self.session = K.get_session()

        self.model = restore_model() if path_to_model is None else restore_model(path_to_model)
        self.class_index_to_human_readable_class = load_classes() if path_to_classes is None else load_classes(path_to_classes)

        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.model_image_width = model_image_width
        self.model_image_height = model_image_height

        self.anchors = anchors / np.array([model_image_width, model_image_height])
        out_tensors, input_tensors = _construct_inference_tensors(
            restored_model=self.model,
            num_of_anchors=3,
            anchors=self.anchors,
            model_image_width=model_image_width,
            model_image_height=model_image_height,
        )

        self.out_tensors = out_tensors
        self.input_tensors = input_tensors
        self.verbose = verbose

    def infer_object_detections_on_loaded_image(
        self,
        image_np: np.array,
    ):
        """
        Infers object detection on the loaded numpy array 
        representing pixels of the image (row-major order).

        :param image_np np.array containing pixels of the image (row-major ordering)
        :return (detected_boxes, detected_classes, detected_scores)
           - detected_boxes array of (left, top, bottom, right)
           - detected_classes array of ints representing class indices
           - detected_scores array of floats representing probability for each box and class
        """
        orig_img_height, orig_img_width = image_np.shape[:2]
        img_np = resize_and_letter_box(image_np / 255., target_width=self.model_image_width, target_height=self.model_image_height)
        img_np = np.expand_dims(img_np, 0)

        start = time.time()
        detected_boxes, detected_classes, detected_scores = infer_objects_in_image(
            image=img_np,
            session=self.session,
            orig_image_height=orig_img_height,
            orig_image_width=orig_img_width,
            out_tensors=self.out_tensors,
            input_tensor=self.input_tensors[0],
            orig_image_width_placeholder_tensor=self.input_tensors[1],
            orig_image_height_placeholder_tensor=self.input_tensors[2]
        )
        end = time.time()
        time_in_s = end - start

        if self.verbose:
            print(f'Took {time_in_s} seconds to run prediction in tf session.')

        return detected_boxes, detected_classes, detected_scores

    def get_mapping_from_class_idx_to_readable_class(self):
        return self.class_index_to_human_readable_class
