from __future__ import print_function

import cv2
import numpy as np
from keras.utils import Sequence
from collections import defaultdict
import sqlite3
import data.openimages.constants as constants
from utils.np_array_db_converters import adapt_array, convert_array
from numpy.random import randint

# Converts numpy array to binary compressed version
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("BLOB", convert_array)
sqlite3.register_converter("blob", convert_array)

PRINT_ENABLED = 1

def debug_print(str, level = 2):
  if level <= PRINT_ENABLED:
    print(str)


class DB:
  def __init__(self, db_images_path, db_image_labels_path):
    self.db_images_path = db_images_path
    self.db_image_labels_path = db_image_labels_path

  def get_positive_image_labels_from(self, original_image_ids):
    original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

    db_image_labels_conn = sqlite3.connect(self.db_image_labels_path, timeout=1000)
    cursor = db_image_labels_conn.cursor()
    cursor.execute("""
      SELECT id, original_image_id, label_id
      FROM image_labels
      WHERE original_image_id IN (%s) AND confidence = 1.0;
    """ % original_image_ids_placeholder, original_image_ids)

    return cursor.fetchall()

  def get_images(self, image_indices, table_name):
    db = sqlite3.connect(self.db_images_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES)
    image_ids_placeholder = ','.join(['?' for _ in range(len(image_indices))])
    cursor = db.cursor()

    get_images_sql = """
      SELECT id, url, original_image_id, image_bytes
      FROM %s
      WHERE id IN (%s);
    """ % (table_name, image_ids_placeholder)

    cursor.execute(get_images_sql, list(map(lambda x: int(x), image_indices)))
    images = cursor.fetchall()

    print(f'get images fetched {len(images)} images.')

    return images



def get_next_batch_indices(total_number_of_samples, batch_size):
  # return list(map(lambda x: int(x), np.random.uniform(1, total_number_of_samples, batch_size)))
  return randint(1, total_number_of_samples, batch_size)


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
    table_name_for_image_urls = 'train_images'
  ):
    self.batch_size = batch_size
    self.table_name_for_image_urls = table_name_for_image_urls
    self.db_images_path = db_images_path
    self.db_image_labels_path = db_image_labels_path

    self.db = DB(db_images_path, db_image_labels_path)

    self.x, self.y = None, None

    self.len = len
    self.num_of_classes = num_of_classes

    self.images_bytes_for_next_batch = []
    self.positive_labels_for_next_batch = []
    self.total_number_of_samples = total_number_of_samples

    self.image_batches = []
    self.next_batch = []

  def __len__(self):
    return self.len

  def get_image_labels_for_images(self, images_from_db):
    original_image_ids = list(set(map(lambda img: img[2], images_from_db)))
    if len(original_image_ids) == 0:
      return []

    print('Getting image labels for image ids: {} ...'.format(', '.join(original_image_ids[:5])))
    positive_image_labels = self.db.get_positive_image_labels_from(original_image_ids)

    original_image_id_to_positive_image_label_ids = defaultdict(list)
    for positive_image_label in positive_image_labels:
      original_image_id_to_positive_image_label_ids[positive_image_label[1]] += [positive_image_label[2]]

    positive_image_labels_for_downloaded_images = []
    for downloaded_image_id in original_image_ids:
      positive_image_labels_for_downloaded_images += [
        original_image_id_to_positive_image_label_ids[downloaded_image_id]]

    return positive_image_labels_for_downloaded_images

  def sample_batch(self):
    indices = get_next_batch_indices(self.total_number_of_samples, self.batch_size)
    results_sample_batch = self.db.get_images(indices, self.table_name_for_image_urls)

    return results_sample_batch


  def prepare_next_batch(self, new_batch_images):
    positive_labels = self.get_image_labels_for_images(new_batch_images)
    self.images_bytes_for_next_batch += list(map(lambda image: image[3], new_batch_images))
    self.positive_labels_for_next_batch += positive_labels


  def ensure_next_batch_is_loaded(self, number_of_batches = 1):
    while len(self.images_bytes_for_next_batch) < (self.batch_size * number_of_batches):
      batch_images_from_db = self.sample_batch()
      self.prepare_next_batch(batch_images_from_db)

      print(f'Length of image bytes for next  batch: {len(self.images_bytes_for_next_batch)}')

    if len(self.images_bytes_for_next_batch) != len(self.positive_labels_for_next_batch):
      raise RuntimeError(f'Number of image bytes prepared != positive labels for next batch.' +
                         f'{images_bytes_for_next_batch} != {self.positive_labels_for_next_batch}')


  def __getitem__(self, idx):
    if len(self.images_bytes_for_next_batch) <= (self.batch_size * 1):
      self.ensure_next_batch_is_loaded(1)

    batch_y = []
    batch_x = []
    for idx, image_bytes in enumerate(self.images_bytes_for_next_batch[:self.batch_size]):
      if image_bytes is None or len(image_bytes) == 0:
        continue
      batch_x += [image_bytes]

      positive_labels_flags = self.positive_labels_for_next_batch[idx]
      y_vector = [1 if i in positive_labels_flags else 0 for i in range(1, constants.Constants.NUM_OF_CLASSES + 1)]

      batch_y += [y_vector]


    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    del self.images_bytes_for_next_batch[:self.batch_size]
    del self.positive_labels_for_next_batch[:self.batch_size]

    print("%s one in batch_y " % (str(len(np.where(batch_y == 1)[0])),))
    print("batch y length %s" %str(len(batch_y)))
    print("%s --> ones" % (str(np.where(batch_y == 1)[1])))
    # print(batch_x)

    return batch_x, batch_y
