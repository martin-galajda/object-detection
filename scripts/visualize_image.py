import argparse
import aiosqlite
import asyncio
import cv2
import numpy as np
import sqlite3
from utils.np_array_db_converters import adapt_array, convert_array

# Converts numpy array to binary compressed version
aiosqlite.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
aiosqlite.register_converter("BLOB", convert_array)
aiosqlite.register_converter("blob", convert_array)
sqlite3.register_converter("blob", convert_array)


def get_positive_image_labels_from_db(original_image_ids, db_labels_path):
  original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

  db_image_labels_conn = sqlite3.connect(db_labels_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute("""
    SELECT image_labels.id, image_labels.original_image_id, image_labels.label_id, labels.label_class_name
    FROM image_labels
    INNER JOIN labels ON image_labels.label_id = labels.id
    WHERE original_image_id IN (%s) AND confidence > 0.0;
  """ % original_image_ids_placeholder, original_image_ids)

  return cursor.fetchall()



async def async_get_image_from_db(image_id, db_path, table_name):
  async with aiosqlite.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
    cursor = await db_conn.execute("""
      SELECT id, original_image_id, url, image_bytes
      FROM %s
      WHERE id = ?;
    """ % table_name, (image_id,))

    db_image = await cursor.fetchone()

    return db_image

async def main(args):
  db_path_images = args.path_to_images_db
  table_name_images = args.table_name_for_images
  db_path_labels = args.path_to_labels_db
  image_id = args.image_id

  image_id, original_image_id, image_url, image_bytes = await async_get_image_from_db(image_id, db_path_images, table_name_images)

  labels = get_positive_image_labels_from_db([original_image_id], db_path_labels)

  print(original_image_id)
  print(list(map(lambda x: x[3], labels)))

  cv2.imshow("Image", image_bytes)
  cv2.waitKey(0)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--path_to_images_db',
    type=str,
    default='./data/openimages/out/db.images.data',
    required=False,
    help='Path to file containing image metadata.')

  parser.add_argument('--path_to_labels_db',
    type=str,
    default='./data/openimages/out/db.labels.data',
    required=False,
    help='Path to file containing labels metadata.')

  parser.add_argument('--table_name_for_images',
    type=str,
    default='train_images',
    required=False,
    help='Name of table which should be appended with image bytes. Default "train_images".')


  parser.add_argument('--image_id',
      type=int,
      default=1,
      required=False,
      help='Image id to visualize.')

  args = parser.parse_args()

  event_loop = asyncio.get_event_loop()
  event_loop.run_until_complete(main(args))