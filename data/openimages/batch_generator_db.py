from __future__ import print_function

import cv2
import numpy as np
from keras.utils import Sequence
from collections import defaultdict
import data.openimages.constants as constants
from utils.np_array_db_converters import adapt_array, convert_array
from numpy.random import randint
from utils.preprocess_image import preprocess_image_bytes
import os
from data.openimages.db import DB

PRINT_ENABLED = 1

placeholder_img_numpy = np.load(constants.Constants.PATH_TO_PLACEHOLDER_NUMPY_225)

def debug_print(str, level=2):
    if level <= PRINT_ENABLED:
        print(str)


def get_next_batch_indices(total_number_of_samples, batch_size):
    # return list(map(lambda x: int(x), np.random.uniform(1, total_number_of_samples, batch_size)))
    return randint(1, total_number_of_samples, batch_size)


def filter_valid_images(images_from_db: list):
    def is_valid_img(img):
        img_bytes = img[3]
        
        if img_bytes is None:
            print(f'Image with id {img[0]} has no bytes ... filtering out')
            return False

        if np.array_equal(placeholder_img_numpy, img_bytes) is True:
            print(f'Image with id {img[0]} is placeholder... filtering out')
            return False
        return True

    return list(filter(is_valid_img, images_from_db))

class OpenImagesData(Sequence):
    """
      Class responsible for loading open images data inside keras training process.
    """

    def __init__(self,
                 batch_size,
                 num_of_classes,
                 len,
                 total_number_of_samples,
                 db_images_path,
                 db_image_labels_path,
                 table_name_for_image_urls='train_images',
                 *,
                 use_multitarget_learning
    ):
        self.seeded = False

        self.batch_size = batch_size
        self.table_name_for_image_urls = table_name_for_image_urls
        self.db_images_path = db_images_path
        self.db_image_labels_path = db_image_labels_path
        self.use_multitarget_learning = use_multitarget_learning

        self.db = DB(db_images_path, db_image_labels_path)

        self.x, self.y = None, None

        self.len = len
        self.num_of_classes = num_of_classes

        self.images_bytes_for_next_batch = []
        self.positive_labels_for_next_batch = []
        self.negative_labels_for_next_batch = []

        self.total_number_of_samples = total_number_of_samples
        self.labels_for_next_batch = []

        self.image_batches = []
        self.next_batch = []

    def __len__(self):
        return self.len

    def get_pos_image_labels_for_images(self, images_from_db):
        original_image_ids = list(map(lambda img: img[2], images_from_db))
        if len(original_image_ids) == 0:
            return []

        # print('Getting image labels for image ids: {} ...'.format(', '.join(original_image_ids[:5])))
        positive_image_labels = self.db.get_positive_image_labels_from(
            original_image_ids)

        original_image_id_to_positive_image_label_ids = defaultdict(list)
        for positive_image_label in positive_image_labels:
            original_image_id_to_positive_image_label_ids[positive_image_label[1]] += [
                positive_image_label[3]]

        positive_image_labels_for_downloaded_images = []
        for downloaded_image_id in original_image_ids:
            positive_image_labels_for_downloaded_images += [
                original_image_id_to_positive_image_label_ids[downloaded_image_id]]

        return positive_image_labels_for_downloaded_images

    def get_image_labels_for_images(self, images_from_db):
        original_image_ids = list(map(lambda img: img[2], images_from_db))
        if len(original_image_ids) == 0:
            return []

        # print('Getting image labels for image ids: {} ...'.format(', '.join(original_image_ids[:5])))
        pos_image_labels, neg_image_labels = self.db.get_trainable_labels_by_original_image_ids(
            original_image_ids)

        original_image_id_to_positive_image_label_ids = defaultdict(list)
        original_image_id_to_negative_image_label_ids = defaultdict(list)

        for positive_image_label in pos_image_labels:
            original_image_id_to_positive_image_label_ids[positive_image_label[1]] += [
                positive_image_label[3]]

        for negative_image_label in neg_image_labels:
            original_image_id_to_negative_image_label_ids[negative_image_label[1]] += [
                negative_image_label[3]]

        positive_image_labels_for_downloaded_images = []
        negative_image_labels_for_downloaded_images = []
        for downloaded_image_id in original_image_ids:
            positive_image_labels_for_downloaded_images += [
                original_image_id_to_positive_image_label_ids[downloaded_image_id]]
            negative_image_labels_for_downloaded_images += [
                original_image_id_to_negative_image_label_ids[downloaded_image_id]
            ]

        return positive_image_labels_for_downloaded_images, negative_image_labels_for_downloaded_images

    def sample_batch(self):
        indices = get_next_batch_indices(
            self.total_number_of_samples, self.batch_size * 2)
        results_sample_batch = self.db.get_images(
            indices, self.table_name_for_image_urls)

        filtered_results_sample_batch = filter_valid_images(results_sample_batch)
        return filtered_results_sample_batch

    def prepare_next_batch(self, new_batch_images):
        if not self.use_multitarget_learning:
            positive_labels = self.get_pos_image_labels_for_images(
                new_batch_images)
            self.positive_labels_for_next_batch += positive_labels
        else:
            pos_labels, neg_labels = self.get_image_labels_for_images(
                new_batch_images)
            self.positive_labels_for_next_batch += pos_labels
            self.negative_labels_for_next_batch += neg_labels

        self.images_bytes_for_next_batch += list(
            map(lambda image: image[3], new_batch_images))

    def ensure_next_batch_is_loaded(self, number_of_batches=1):
        while len(self.images_bytes_for_next_batch) < (self.batch_size * number_of_batches):
            batch_images_from_db = self.sample_batch()
            self.prepare_next_batch(batch_images_from_db)

            print(
                f'Length of image bytes for next batch: {len(self.images_bytes_for_next_batch)}')

        if len(self.images_bytes_for_next_batch) != len(self.positive_labels_for_next_batch):
            raise RuntimeError(f'Number of image bytes prepared != positive labels for next batch.' +
                               f'{self.images_bytes_for_next_batch} != {self.positive_labels_for_next_batch}')

    def __getitem__(self, idx):
        if not self.seeded:
            # as we use multiprocessing - we have to make randomness different in all processes
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            self.seeded = True

        if len(self.images_bytes_for_next_batch) <= (self.batch_size * 1):
            self.ensure_next_batch_is_loaded(1)

        batch_y = []
        batch_x = []
        for idx, image_bytes in enumerate(self.images_bytes_for_next_batch[:self.batch_size]):
            if image_bytes is None or len(image_bytes) == 0:
                continue
            batch_x += [preprocess_image_bytes(image_bytes)]

            # print(f'Shape self.positive_labels_for_next_batch: {np.array(self.positive_labels_for_next_batch).shape}, idx: {idx}')
            positive_labels_flags = self.positive_labels_for_next_batch[idx]

            if not self.use_multitarget_learning:
                y_vector = [1 if i in positive_labels_flags else 0 for i in range(
                    1, constants.Constants.NUM_OF_TRAINABLE_CLASSES + 1)]
            else:
                y_vector = np.repeat(constants.Constants.MASK_VALUE_MISSING_LABEL,
                                     constants.Constants.NUM_OF_TRAINABLE_CLASSES)

                # in db label ids start from 1, but we index our array starting from 0 so just shift value by one to the left
                positive_labels_flags = self.positive_labels_for_next_batch[idx] - np.array([
                                                                                            1], dtype=np.int32)
                negative_labels_flags = self.negative_labels_for_next_batch[idx] - np.array([
                                                                                            1], dtype=np.int32)

                y_vector[np.array(positive_labels_flags, dtype=np.int32)] = 1.0
                y_vector[np.array(negative_labels_flags, dtype=np.int32)] = 0.0

            batch_y += [y_vector]

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        del self.images_bytes_for_next_batch[:self.batch_size]
        del self.positive_labels_for_next_batch[:self.batch_size]

        # print("%s one in batch_y " % (str(len(np.where(batch_y == 1)[0])),))
        # print("batch y length %s" %str(len(batch_y)))
        # print("%s --> ones" % (str(len(np.where(batch_y == 1)[1]))))
        # print("%s --> dummies" % (str(len(np.where(batch_y == -1)[1]))))
        # print("%s --> negatives" % (str(len(np.where(batch_y == 0)[1]))))
        # print(batch_x)

        return batch_x, batch_y
