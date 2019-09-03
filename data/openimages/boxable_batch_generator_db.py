from __future__ import print_function

import numpy as np
import os
import asyncio

from common.assertions import assert_true
from keras.utils import Sequence
from utils.array_operations import array_intersection
from utils.preprocess_image import preprocess_image_bytes
from data.openimages.constants import BoxableImagesConstants
from utils.sampling import sample_values
from models.yolov2.utils.load_anchors import load_anchors
from models.yolov2.preprocessing.training import preprocess_image_bytes, preprocess_true_boxes, get_detector_mask
from data.openimages.boxable_db import async_get_boxes_by_image_ids, \
        async_get_images_by_ids, \
        get_non_empty_boxes


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()

    return loop


async def get_non_empty_images(
    *,
    db_path: str,
    table_name_images: str,
    image_ids: list,
):
    images = await async_get_images_by_ids(
        db_path=db_path,
        table_name_images=table_name_images,
        image_ids=image_ids,
        only_non_empty=True
    )
    placeholder_img_numpy = np.load(BoxableImagesConstants.BOXABLE_PLACEHOLDER_PATH)

    def is_not_placeholder(img):
        if not np.array_equal(np.array(img[1], dtype=np.uint8),  placeholder_img_numpy):
            return True

        print(f'Img with id {img[0]} is placeholder. Filtering out')
        return False

    return filter(is_not_placeholder, images)


async def async_get_images_and_boxes_for_batch(
    *,
    db_path: str,
    table_name_images: str,
    table_name_boxes: str,
    image_ids: list,
    required_batch_size: int
):
    non_empty_images_future = async_get_images_by_ids(
        db_path=db_path,
        table_name_images=table_name_images,
        image_ids=image_ids,
        only_non_empty=True,
    )

    non_empty_boxes_future = get_non_empty_boxes(
        db_path=db_path,
        table_name_boxes=table_name_boxes,
        sampled_image_ids=image_ids,
        required_batch=required_batch_size,
        only_positive=True
    )

    non_empty_images, non_empty_boxes_res = await asyncio.gather(non_empty_images_future, non_empty_boxes_future)

    # 1 is index of col retrieved with name "image_id"
    # non_empty_boxes_image_ids = [box[1] \
    #                              # iterate over boxes for all images that have at least one box
    #                              # (array of arrays)
    #                              for boxes_for_image in non_empty_boxes \
    #                              # we need only first box for the image (as all have same image ids)
    #                              for box in boxes_for_image[:1]]

    non_empty_boxes = non_empty_boxes_res[0]
    def get_image_ids_from_boxes(boxes_for_images: list):
        ids = []
        for boxes_for_image in boxes_for_images:
            first_box = boxes_for_image[0]
            ids += [first_box[1]]

        return ids
    non_empty_boxes_image_ids = get_image_ids_from_boxes(non_empty_boxes)

    # 0 is index of col retrieved with name "id" ~> "image_id"
    non_empty_images_image_ids = [img[0] for img in non_empty_images]

    image_ids_intersection = array_intersection(non_empty_images_image_ids, non_empty_boxes_image_ids)

    result_images = [img for img in non_empty_images if img[0] in image_ids_intersection]

    # take first box(index 0) and use it's image_id column(index 1)
    result_boxes = list(filter(lambda boxes_for_image: boxes_for_image[0][1] in image_ids_intersection, non_empty_boxes))

    return result_images, result_boxes


def get_images_and_boxes_for_batch(
    *,
    db_path: str,
    table_name_images: str,
    table_name_boxes: str,
    image_ids: list,
    required_batch_size: int
):
    event_loop = get_event_loop()
    images, boxes_for_batch = event_loop.run_until_complete(async_get_images_and_boxes_for_batch(
        db_path=db_path,
        table_name_images=table_name_images,
        table_name_boxes=table_name_boxes,
        image_ids=image_ids,
        required_batch_size=required_batch_size
    ))

    return images, boxes_for_batch


def has_valid_img_bytes(img_bytes):
    return img_bytes is not None


def zeropad_boxes(batch_image_boxes, max_num_of_boxes_in_batch):
    zeropadded_results = []
    for batch_images in batch_image_boxes:
        len_difference_to_zeropad = max_num_of_boxes_in_batch - len(batch_images)

        if len_difference_to_zeropad > 0:
            new_zeropadded = np.zeros(len_difference_to_zeropad * 5).reshape((len_difference_to_zeropad, 5))
            new_results = batch_images + list(new_zeropadded)
            zeropadded_results += [new_results]
        else:
            zeropadded_results += [batch_images]

    return np.array(zeropadded_results)


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
        table_name_for_images=BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES,
        table_name_for_image_boxes=BoxableImagesConstants.TABLE_NAME_TRAIN_IMAGE_BOXES,
        num_of_classes=BoxableImagesConstants.NUM_OF_CLASSES,
    ):
        self.seeded = False

        self.db_path = db_path
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
            min_value=1,
            max_value=self.total_number_of_samples,
            batch_size=self.batch_size * 4
        )

        images, boxes_for_batch = get_images_and_boxes_for_batch(
            db_path=self.db_path,
            table_name_images=self.table_name_for_images,
            table_name_boxes=self.table_name_for_image_boxes,
            image_ids=indices,
            required_batch_size=self.batch_size
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

        assert len(preprocessed_image_bytes) == len(matching_true_boxes), \
            'len(preprocessed_image_bytes) != len(matching_true_boxes)'
        assert len(preprocessed_image_boxes) == len(detectors_mask), \
            'len(preprocessed_image_boxes) != len(detectors_mask)'
        assert len(preprocessed_image_bytes) == len(detectors_mask), \
            'len(preprocessed_image_bytes) != len(detectors_mask)'

        return preprocessed_image_bytes, preprocessed_image_boxes

    def __getitem__(self, idx):
        if not self.seeded:
            # as we use multiprocessing - we have to make randomness different in all processes
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            self.seeded = True

        batch_x, batch_y = self.sample_batch()

        return batch_x, batch_y
