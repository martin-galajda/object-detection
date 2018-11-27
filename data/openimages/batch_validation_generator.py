from __future__ import print_function

from keras.utils import Sequence
import sys
import asyncio
from multiprocessing import Pool, Queue
import aiohttp
import cv2
import numpy as np
import os
from  time import time as timer
from itertools import islice
from keras.utils import Sequence
from time import sleep
from collections import defaultdict
import sqlite3
import data.openimages.constants as constants
import aiosqlite

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
    WHERE original_image_id IN (%s);
  """ % original_image_ids_placeholder, original_image_ids)

  return cursor.fetchall()

def get_next_batch_indices(total_number_of_samples, batch_size):
  return map(lambda x: int(x), np.random.uniform(0, total_number_of_samples, batch_size))

async def get_image_urls_from_db(image_indices):
  IMAGES_DB_PATH = './data/openimages/db.images.data'

  try:
    async with aiosqlite.connect(IMAGES_DB_PATH, timeout=1000) as db:
      image_ids_placeholder = ','.join(['?' for _ in range(len(image_indices))])
      await db.execute("""
        SELECT id, url, original_image_id
        FROM train_images
        WHERE id IN (%s);
      """ % image_ids_placeholder, image_indices)

      return db.fetchall()
  except Exception as e:
    print("Error occured in get_image_urls_from_db(image_indices): " + str(e))

def get_image_labels_from_images(downloaded_image_tuples):
  image_ids = list(set(map(lambda img: img[1], downloaded_image_tuples[0])))
  if len(image_ids) == 0:
    return []

  positive_image_labels = get_positive_image_labels_from_db(image_ids)

  original_image_id_to_positive_image_label_ids = defaultdict(list)
  for positive_image_label in positive_image_labels:
    original_image_id_to_positive_image_label_ids[positive_image_label[1]] += [positive_image_label[2]]


  positive_image_labels_for_downloaded_images = []
  for downloaded_image_id in image_ids:
    positive_image_labels_for_downloaded_images += [original_image_id_to_positive_image_label_ids[downloaded_image_id]]


  return positive_image_labels_for_downloaded_images


# def prefetch_train_data(batch_size, num_of_instances):
def download_images_by_urls_and_ids(urls_and_ids):
  try:
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    loop = asyncio.get_event_loop()
    downloaded_image_content_and_id_tuples = loop.run_until_complete(asyncio.gather(async_download_images(urls_and_ids)))
    positive_image_labels = get_image_labels_from_images(downloaded_image_content_and_id_tuples)
    return (downloaded_image_content_and_id_tuples, positive_image_labels)
  except Exception as e:
    print("Exception occured : %s!" % str(e))



def init_worker(_queue):
  ''' store the queue for later use '''
  global global_queue
  global_queue = _queue


MAX_PENDING_WORKERS = 3
PRELOADED_BATCHES_COUNT = 5


class OpenImagesData(Sequence):
  def __init__(self, file_train_image_urls, batch_size, num_of_classes, len, total_number_of_samples):
    self.batch_size = batch_size
    self.last_img_idx = 0

    self.queue = Queue()
    self.workers_pool = Pool(initializer = init_worker, initargs = (self.queue, ), processes=10)

    self.x, self.y = None, None

    self.len = len
    self.file_train_image_urls = file_train_image_urls
    self.num_of_classes = num_of_classes

    self.worker_tasks = []
    self.images_for_next_batch = []
    self.positive_labels_for_next_batch = []
    self.total_number_of_samples = total_number_of_samples

    self.image_batches = []
    self.loop = asyncio.get_event_loop()
    self.loop.run_until_complete(asyncio.gather(self.sample_batch()))
    self.loop.run_forever()

    [asyncio.ensure_future(self.sample_batch(), self.loop) for _ in range(PRELOADED_BATCHES_COUNT)]

    self.prefetch_new_images(1)

  def __len__(self):
    return self.len

  async def sample_batch(self):
    indices = get_next_batch_indices(self.total_number_of_samples, self.batch_size)

    self.image_batches += [await get_image_urls_from_db(indices)]

  def prefetch_new_images(self, times):
    for _ in range(times):
      self.download_images()

  def download_images(self):
    print("Going to download images using pool...")
    num_of_batch_downloads = 10
    batch_download_size = self.batch_size / num_of_batch_downloads

    new_batch_images = self.image_batches[0]
    del self.image_batches[0]

    worker_data_payload = []
    for img_from_batch in new_batch_images:
     # id, url, original_image_id
      img_original_id = img_from_batch[2]
      img_url = img_from_batch[1]

      worker_data_payload += [(img_url, img_original_id)]

      if len(worker_data_payload) > 0 and ((len(worker_data_payload) % batch_download_size) == 0):
        self.worker_tasks += [self.workers_pool.map_async(download_images_by_urls_and_ids, (worker_data_payload,))]
        worker_data_payload = []

    # with open(self.file_train_image_urls) as tsv_file:
    #   lines = []
    #   slice = islice(tsv_file, self.last_img_idx, self.last_img_idx + (num_of_batch_downloads * batch_download_size))
    #
    #   for line in slice:
    #
    #     lines += [line]
    #
    #     if len(lines) > 0 and len(lines) % batch_download_size == 0:
    #       lines = []
    #
    #       self.worker_tasks += [self.workers_pool.map_async(download_images_by_urls_and_ids, (lines,))]
    #
    #   self.last_img_idx += int(num_of_batch_downloads * batch_download_size)
    #
    #   if self.last_img_idx >= 1000000:
    #     self.last_img_idx = 0

  def __getitem__(self, idx):
    if len(self.image_batches) == 0:
      print("Not a single image_batch (urls of images from db), running sample_batch until complete")
      self.loop.run_until_complete(asyncio.gather(self.sample_batch()))

    if len(self.image_batches) < PRELOADED_BATCHES_COUNT:
      print("Not enough image_batches preloaded... adding async function to preload it")
      asyncio.ensure_future(self.sample_batch(), self.loop)


    if len(self.images_for_next_batch) <= (self.batch_size * 10):
      print("__getitem__ going to execute new download image")
      self.prefetch_new_images(1)
      print("__getitem__ after executing new download image")

    while len(self.images_for_next_batch) < self.batch_size:
      if len(self.worker_tasks) == 0:
        self.prefetch_new_images(1)

      some_worker_was_ready = False

      worker_indices_finished = []

      for idx, worker_task in enumerate(self.worker_tasks):
        if worker_task.ready():
          worker_results = worker_task.get()[0]
          print("worker result shape: ")
          print(np.array(worker_results).shape)
          self.images_for_next_batch += worker_results[0][0]
          self.positive_labels_for_next_batch += worker_results[1]

          print("len(self.images_for_next_batch): " + str(len(self.images_for_next_batch)))
          print("len(self.positive_labels_for_next_batch): " + str(len(self.positive_labels_for_next_batch)))

          some_worker_was_ready = True

          worker_indices_finished += [idx]

      self.worker_tasks = \
        [
          worker_task
           for worker_task_idx, worker_task
           in enumerate(self.worker_tasks)
           if worker_task_idx not in worker_indices_finished
        ]


      if not some_worker_was_ready:
        print("Not a single worker ready... sleeping")
        sleep(3)
        continue

      if len(self.worker_tasks) < 3:
        self.prefetch_new_images(3)

    batch_y = []
    batch_x = []
    for idx, image in enumerate(self.images_for_next_batch[:self.batch_size]):
      image_data = image[0]
      batch_x += [image_data]

      positive_labels_flags = self.positive_labels_for_next_batch[idx]
      y_vector = [1 if i in positive_labels_flags else 0 for i in range(1, constants.Constants.NUM_OF_CLASSES + 1)]
      batch_y += [y_vector]


    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    del self.images_for_next_batch[:self.batch_size]
    del self.positive_labels_for_next_batch[:self.batch_size]

    print("batch_x shape: " + str(batch_x.shape))
    print("batch_y shape: " + str(batch_y.shape))

    return batch_x, batch_y


  def __del__(self):
    self.workers_pool.close()
    self.workers_pool.join()
