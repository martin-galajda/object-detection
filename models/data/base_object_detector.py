from abc import ABC, abstractmethod
from models.data.bounding_box import BoundingBox, make_bounding_boxes
from typing import List, Dict
import numpy as np
from utils.image import load_pil_image_from_file


class BaseObjectDetector(ABC):
    """
    Class responsible for producing object detections on images that can be loaded:
        - via numpy.array representing RGB pixels of image
        - valid filepath to the image on the current filesystem

    Methods provided:
     - infer_object_detections_on_loaded_image - produce raw object detections on numpy array
     - infer_object_detections - produce raw object detections on file
     - infer_bounding_boxes_on_loaded_image - produce object detections in form
        of List of BoundingBox classes on numpy array
     - infer_bounding_boxes_on_target_path - produce object detections in form
        of List of BoundingBox classes on image located by file path
     - close: Clean-up method, call when done with working with object detector.

    """

    @abstractmethod
    def infer_object_detections_on_loaded_image(
        self,
        image_np: np.array,
    ) -> tuple:
        """
        Method responsible for inferring object detections on numpy array
        representing image - detections should be produced in form of
        (inferred_boxes, inferred_class_indices, inferred_scores).

        :param image_np: numpy array representing image
        :return: (inferred_boxes, inferred_class_indices, inferred_scores) where
            - inferred_boxes is iterable of (left, top, right, bottom) point of bounding box
            - inferred_class_indices is iterable of integer indices for classes
            - inferred_scores is iterable of float scores representing probability for given class located
                in the image
        """
        pass

    @abstractmethod
    def get_mapping_from_class_idx_to_readable_class(self) -> Dict[int, str]:
        pass

    def infer_object_detections(self, target_file_path: str) -> tuple:
        """
        Method responsible for loading image at 'target_file_path' and
        inferring object detections in form of
        (inferred_boxes, inferred_class_indices, inferred_scores).

        :param target_file_path: path to file on which run object detection.
        :return: (inferred_boxes, inferred_class_indices, inferred_scores) where
            - inferred_boxes is iterable of (left, top, right, bottom) point of bounding box
            - inferred_class_indices is iterable of integer indices for classes
            - inferred_scores is iterable of float scores representing probability for given class located
                in the image
        """
        _, img_np = load_pil_image_from_file(target_file_path)
        return self.infer_object_detections_on_loaded_image(img_np)

    def close(self) -> None:
        """
        Optional method for closing resources allocated by the object detector.
        """
        pass

    def infer_bounding_boxes_on_loaded_image(
        self,
        image_np: np.array
    ) -> List[BoundingBox]:
        """
        Method responsible for inferring object detections on numpy array
        representing image.

        Returns List of Bounding Boxes for easier manipulation.

        :param image_np: numpy array representing image
        :return: list of BoundingBox objects
        """
        boxes, classes, scores = self.infer_object_detections_on_loaded_image(image_np)

        return self._inference_outputs_to_bounding_boxes(
            boxes=boxes,
            classes=classes,
            scores=scores
        )

    def infer_bounding_boxes_on_target_path(self, target_file_path: str) -> List[BoundingBox]:
        """
        Method responsible for inferring object detections on image located
        by path specified by 'target_file_path' on current filesystem.

        Returns List of Bounding Boxes for easier manipulation.

        :param target_file_path: numpy array representing image
        :return: list of BoundingBox objects
        """

        boxes, classes, scores = self.infer_object_detections(target_file_path)

        return self._inference_outputs_to_bounding_boxes(
            boxes=boxes,
            classes=classes,
            scores=scores
        )

    def _inference_outputs_to_bounding_boxes(
        self,
        *,
        boxes,
        classes,
        scores
    ):
        """
        Private helper method for mapping inference outputs to List of BoundingBox objects.
        """
        return make_bounding_boxes(
            inferred_boxes=boxes,
            inferred_classes=classes,
            inferred_scores=scores,
            class_index_to_human_readable_dict=self.get_mapping_from_class_idx_to_readable_class()
        )
