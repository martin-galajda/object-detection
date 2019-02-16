from __future__ import print_function

import cv2
import numpy as np
import os
import asyncio

from keras.utils import Sequence
from collections import defaultdict
import data.openimages.constants as constants
from utils.np_array_db_converters import adapt_array, convert_array
from numpy.random import randint
from utils.preprocess_image import preprocess_image_bytes
from data.openimages.db import DB
from data.openimages.constants import BoxableImagesConstants
from utils.sampling import sample_values
from models.yolov2.utils.load_anchors import load_anchors
from models.yolov2.preprocessing.training import preprocess_image_bytes, preprocess_true_boxes, get_detector_mask
from data.openimages.boxable_db import async_get_boxes_by_image_ids, async_get_images_by_ids

PRINT_ENABLED = 1


def debug_print(str, level = 2):
    if level <= PRINT_ENABLED:
        print(str)


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()

    return loop


async def async_get_images_and_boxes_for_batch(
    *,
    db_path: str,
    table_name_images: str,
    table_name_boxes: str,
    image_ids: list
):
    get_images_future = async_get_images_by_ids(
        db_path=db_path,
        table_name_images=table_name_images,
        image_ids=image_ids)
    get_boxes_future = async_get_boxes_by_image_ids(
        db_path=db_path,
        table_name_boxes=table_name_boxes,
        image_ids=image_ids
    )

    result = await asyncio.gather(get_images_future, get_boxes_future)

    return result


def get_images_and_boxes_for_batch(
    *,
    db_path: str,
    table_name_images: str,
    table_name_boxes: str,
    image_ids: list
):
    event_loop = get_event_loop()
    images, boxes_for_batch = event_loop.run_until_complete(async_get_images_and_boxes_for_batch(
        db_path=db_path,
        table_name_images=table_name_images,
        table_name_boxes=table_name_boxes,
        image_ids=image_ids
    ))

    return images, boxes_for_batch


def has_valid_img_bytes(img_bytes):
    return img_bytes is not None


class BoxableOpenImagesData(Sequence):
    """
      Class responsible for loading open images data inside keras training process.
    """

    def __init__(
        self,
        batch_size,
        len,
        total_number_of_samples,
        *,
        use_multitarget_learning,
        db_path,
        table_name_for_images = BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES,
        table_name_for_image_boxes = BoxableImagesConstants.TABLE_NAME_TRAIN_IMAGE_BOXES,
        num_of_classes = BoxableImagesConstants.NUM_OF_CLASSES,
    ):
        self.seeded = False

        self.db_path  = db_path
        self.table_name_for_images = table_name_for_images
        self.table_name_for_image_boxes = table_name_for_image_boxes

        self.batch_size = batch_size
        self.use_multitarget_learning = use_multitarget_learning

        self.x, self.y = None, None

        self.len = len
        self.num_of_classes = num_of_classes

        self.images_bytes_for_next_batch = []
        self.positive_labels_for_next_batch = []
        self.negative_labels_for_next_batch = []

        self.total_number_of_samples = total_number_of_samples

        self.image_bytes_for_next_batch = []
        self.boxes_for_next_batch = []
        self.detector_mask_for_next_batch = []
        self.matching_boxes_for_next_batch = []

    def __len__(self):
        return self.len

    def sample_batch(self):
        indices = sample_values(
            min_value = 1,
            max_value = self.total_number_of_samples,
            batch_size = self.batch_size * 1
        )

        indices = [int(i) for i in indices]

        images, boxes_for_batch = get_images_and_boxes_for_batch(
            db_path=self.db_path,
            table_name_images=self.table_name_for_images,
            table_name_boxes=self.table_name_for_image_boxes,
            image_ids=indices
        )

        def map_box(box):
            label_id, x_min, x_max, y_min, y_max = box[2:7]
            return np.array([label_id, x_min, y_min, x_max, y_max])

        prepared_boxes = []
        batch_image_bytes = []

        for img_idx, image in enumerate(images):
            img_bytes = image[1]

            if not has_valid_img_bytes(img_bytes):
                continue

            batch_image_bytes += [img_bytes]
            boxes_for_img = boxes_for_batch[img_idx]
            prepared_boxes += [np.array(list(map(map_box, boxes_for_img)))]

        prepared_boxes = np.array(prepared_boxes)

        print(f'Shape: {np.array(batch_image_bytes).shape}')
        preprocessed_image_bytes = preprocess_image_bytes(batch_image_bytes)
        preprocessed_image_boxes = preprocess_true_boxes(prepared_boxes)

        anchors = load_anchors()
        detectors_mask, matching_true_boxes = get_detector_mask(preprocessed_image_boxes, anchors)

        return [preprocessed_image_bytes, preprocessed_image_boxes, detectors_mask, matching_true_boxes]

    def __getitem__(self, idx):
        if not self.seeded:
            # as we use multiprocessing - we have to make randomness different in all processes
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            self.seeded = True

        batch_x = self.sample_batch()
        batch_y = np.zeros(len(batch_x[0]))

        return batch_x, batch_y
