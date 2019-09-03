from __future__ import print_function

import sys
import asyncio
import aiohttp
import cv2
import numpy as np
from keras.utils import Sequence
from collections import defaultdict
import sqlite3
import data.openimages.constants as constants
import aiosqlite

PRINT_ENABLED = 1


def debug_print(str, level = 2):
    if level <= PRINT_ENABLED:
        print(str)

async def download_data_from_url(session, url):
    try:
        async with session.get(url) as response:
            response_text = await response.read()

            return response_text
    except Exception as exc:
        print("Error in download_data_from_url(session, url): " + str(exc), file = sys.stderr)

async def download_image(url, id):
    try:
        async with aiohttp.ClientSession() as session:
            bytes_img = await download_data_from_url(session, url)

            img_numpy = np.fromstring(bytes_img, np.uint8)

            img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

            img_resized = cv2.resize(img_cv2, (224, 224))
            img = img_resized.astype(np.uint8)

            return (img, id)
    except Exception as e:
        print("Error in download_image(url, id):  " + str(e), file = sys.stderr)

async def async_download_images(urls_and_ids):
    try:
        futures = []
        for url_and_original_img_id in urls_and_ids:
            image_url, image_original_id = url_and_original_img_id

            futures += [download_image(image_url, image_original_id)]

        sql_values_for_db_update = await asyncio.gather(*futures)
        debug_print("Downloaded %d images." % len(sql_values_for_db_update), level = 2)
        return sql_values_for_db_update
    except Exception as e:
        print("Error downloading and saving images " + str(e))


def get_positive_image_labels_from_db(original_image_ids):
    original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

    db_image_labels_conn = sqlite3.connect(constants.Constants.IMAGE_LABELS_DB_PATH)
    cursor = db_image_labels_conn.cursor()
    cursor.execute("""
    SELECT id, original_image_id, label_id
    FROM image_labels
    WHERE original_image_id IN (%s) AND confidence = 1.0;
  """ % original_image_ids_placeholder, original_image_ids)

    return cursor.fetchall()

def get_next_batch_indices(total_number_of_samples, batch_size):
    return list(map(lambda x: int(x), np.random.uniform(0, total_number_of_samples, batch_size)))

async def get_image_urls_from_db(image_indices, table_name):
    try:
        async with aiosqlite.connect(constants.Constants.IMAGES_DB_PATH, timeout=1000) as db:
            image_ids_placeholder = ','.join(['?' for _ in range(len(image_indices))])
            cursor = await db.cursor()
            await cursor.execute("""
        SELECT id, url, original_image_id
        FROM '%s'
        WHERE id IN (%s);
      """ % (table_name, image_ids_placeholder), image_indices)

            return await cursor.fetchall()
    except Exception as e:
        print("Error occured in get_image_urls_from_db(image_indices): " + str(e), file = sys.stderr)

def get_image_labels_from_images(downloaded_image_tuples):
    image_ids = list(set(map(lambda img: img[1], downloaded_image_tuples)))
    if len(image_ids) == 0:
        return []

    positive_image_labels = get_positive_image_labels_from_db(image_ids)

    original_image_id_to_positive_image_label_ids = defaultdict(list)
    for positive_image_label in positive_image_labels:
        original_image_id_to_positive_image_label_ids[positive_image_label[1]] += [positive_image_label[2]]


    # print("original_image_id_to_positive_image_label_ids:")
    # print(original_image_id_to_positive_image_label_ids)

    positive_image_labels_for_downloaded_images = []
    for downloaded_image_id in image_ids:
        positive_image_labels_for_downloaded_images += [original_image_id_to_positive_image_label_ids[downloaded_image_id]]

    # print("positive_image_labels_for_downloaded_images:")
    # print(positive_image_labels_for_downloaded_images)
    return positive_image_labels_for_downloaded_images


# def prefetch_train_data(batch_size, num_of_instances):
def download_images_by_urls_and_ids(urls_and_ids):

    try:
        loop = asyncio.get_event_loop()
    except:
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()

    try:
        # policy = asyncio.get_event_loop_policy()
        # policy.set_event_loop(policy.new_event_loop())
        # loop = asyncio.get_event_loop()
        downloaded_image_content_and_id_tuples = loop.run_until_complete(async_download_images(urls_and_ids))
        positive_image_labels = get_image_labels_from_images(downloaded_image_content_and_id_tuples)
        return (downloaded_image_content_and_id_tuples, positive_image_labels)
    except Exception as e:
        print("Exception occured : %s!" % str(e), file = sys.stderr)

def get_image_urls_from_db_sync(image_indices, table_name):
    # policy = asyncio.get_event_loop_policy()
    # policy.set_event_loop(policy.new_event_loop())
    # loop = asyncio.get_event_loop()

    try:
        loop = asyncio.get_event_loop()
    except:
        policy = asyncio.get_event_loop_policy()
        policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.get_event_loop()

    try:
        debug_print("Getting image indices from table name %s " % table_name)
        image_urls_with_ids = loop.run_until_complete(get_image_urls_from_db(image_indices, table_name))
        return image_urls_with_ids
    except Exception as e:
        print("Exception occured in get_image_urls_from_db_sync() : %s!" % str(e), file=sys.stderr)


class OpenImagesData(Sequence):
    def __init__(self,
                 batch_size,
                 num_of_classes,
                 len,
                 total_number_of_samples,
                 table_name_for_image_urls = 'train_images'
                 ):
        self.batch_size = batch_size
        self.table_name_for_image_urls = table_name_for_image_urls

        self.x, self.y = None, None

        self.len = len
        self.num_of_classes = num_of_classes

        self.images_bytes_for_next_batch = []
        self.positive_labels_for_next_batch = []
        self.total_number_of_samples = total_number_of_samples

        self.image_batches = []

    def __len__(self):
        return self.len

    def sample_batch(self):
        debug_print("Sampling batch (getting imgs from db)")
        indices = get_next_batch_indices(self.total_number_of_samples, self.batch_size)
        results_sample_batch = get_image_urls_from_db_sync(indices, self.table_name_for_image_urls)

        self.image_batches += [results_sample_batch]

    def download_images(self):
        debug_print("Going to download images...")
        batch_download_size = self.batch_size

        new_batch_images = self.image_batches[0]
        del self.image_batches[0]

        worker_data_payload = []

        debug_print("new_batch_images length: %d." % len(new_batch_images))
        for img_from_batch in new_batch_images:
            img_original_id = img_from_batch[2]
            img_url = img_from_batch[1]

            worker_data_payload += [(img_url, img_original_id)]

            if len(worker_data_payload) > 0 and ((len(worker_data_payload) % batch_download_size) == 0):
                download_results = download_images_by_urls_and_ids(worker_data_payload)

                debug_print("download_results shape: %s" % str(np.array(download_results).shape))
                if download_results and download_results[0] and download_results[1]:
                    self.images_bytes_for_next_batch += download_results[0]
                    self.positive_labels_for_next_batch += download_results[1]

                    debug_print("self.images_bytes_for_next_batch shape: %s" % str(np.array(self.images_bytes_for_next_batch).shape))

                worker_data_payload = []

    def ensure_next_batch_is_loaded(self, number_of_batches = 1):
        while len(self.images_bytes_for_next_batch) < (self.batch_size * number_of_batches):
            self.sample_batch()
            self.download_images()

    def __getitem__(self, idx):
        if len(self.images_bytes_for_next_batch) <= (self.batch_size * 3):
            debug_print("__getitem__ going to execute new download image")
            self.ensure_next_batch_is_loaded(3)
            debug_print("__getitem__ after executing new download image")

        batch_y = []
        batch_x = []
        for idx, image in enumerate(self.images_bytes_for_next_batch[:self.batch_size]):
            image_data = image[0]
            batch_x += [image_data]

            positive_labels_flags = self.positive_labels_for_next_batch[idx]
            y_vector = [1 if i in positive_labels_flags else 0 for i in range(1, constants.Constants.NUM_OF_CLASSES + 1)]

            # print("Adding %d positive classes" % len(np.where(np.array(y_vector) > 0)[0]))
            batch_y += [y_vector]

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        del self.images_bytes_for_next_batch[:self.batch_size]
        del self.positive_labels_for_next_batch[:self.batch_size]

        debug_print("self.images_bytes_for_next_batch shape: %s" % str(np.array(self.images_bytes_for_next_batch).shape))
        # print("batch_x shape: " + str(batch_x.shape))
        # print("batch_y shape: " + str(batch_y.shape))
        print("%s one in batch_y " % (str(len(np.where(batch_y == 1)[0])),))
        print("batch y length %s" %str(len(batch_y)))
        print(batch_y)

        return batch_x, batch_y
