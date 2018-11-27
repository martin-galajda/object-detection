import sqlite3
import argparse
import re
import csv
from time import time as timer

DB_PATH = './data/openimages/out/db.images.data'

TRAIN_TABLE_NAME = 'train_images'
VALIDATION_TABLE_NAME = 'validation_images'

DEFAULT_PATH_TO_TRAIN_IMAGES_CSV = './data/openimages/train-images-with-labels-with-rotation.csv'
DEFAULT_PATH_TO_VALIDATION_IMAGES_CSV = './data/openimages/validation-images-with-rotation.csv'

def setup_images_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  cursor.executescript("""
    CREATE TABLE IF NOT EXISTS %s (
      id INTEGER PRIMARY KEY NOT NULL,
      original_image_id VARCHAR NOT NULL UNIQUE,
      url VARCHAR NOT NULL,
      rotation REAL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,

      UNIQUE (url, rotation)
    );

    CREATE TABLE IF NOT EXISTS %s (
      id INTEGER PRIMARY KEY NOT NULL,
      original_image_id VARCHAR NOT NULL UNIQUE,
      url VARCHAR NOT NULL,
      rotation REAL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,

      UNIQUE (url, rotation)
    );

  """ % (TRAIN_TABLE_NAME, VALIDATION_TABLE_NAME))
  conn.commit()


def save_images_into_db(db_table_name, images_with_urls):
  conn = sqlite3.connect(DB_PATH, timeout=500)
  cursor = conn.cursor()

  cursor.executemany("""
    INSERT into %s (original_image_id, url)
    VALUES (?, ?);
  """ % db_table_name, images_with_urls)

  print("Inserted %d images with urls into db." % cursor.rowcount)

  conn.commit()


def import_images_with_urls_to_db(path_to_annotations_csv_file, db_table_name):
  batch_update_size = 50000
  images_with_urls = []
  start = timer()
  processed = 0

  with open(path_to_annotations_csv_file, 'r', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if not row['Thumbnail300KURL']:
        row['Thumbnail300KURL'] = row['OriginalURL']
      images_with_urls += [(row['ImageID'], row['Thumbnail300KURL'])]

      if len(images_with_urls) > 0 and ((len(images_with_urls) % batch_update_size) == 0):
        save_images_into_db(db_table_name, images_with_urls)
        processed += len(images_with_urls)
        images_with_urls = []


      if len(images_with_urls) == 0 and processed > 0 and (processed % (batch_update_size * 5) == 0):
        print("Processed %d images with  urls." % processed)
        now = timer()
        time_from_start = now - start
        process_one_takes_time = time_from_start / processed
        print("Took %f seconds to insert %d images" % (time_from_start, processed))

        estimated_time_of_finishing_in_s = process_one_takes_time * (5500000 - processed)
        estimated_time_of_finishing_in_m = estimated_time_of_finishing_in_s / 60
        print("Estimated time of finishing: %f seconds " % estimated_time_of_finishing_in_s)
        print("Estimated time of finishing: %f minutes " % estimated_time_of_finishing_in_m)

  if len(images_with_urls) > 0:
    save_images_into_db(db_table_name, images_with_urls)
    processed += len(images_with_urls)

  print("Processed %d images with urls." % processed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--path_to_db',
    type=str,
    default=DB_PATH,
    required=False,
    help='Path to database file to which import image ids with urls.'
  )

  parser.add_argument('--path_to_train_images_csv',
    type=str,
    default=DEFAULT_PATH_TO_TRAIN_IMAGES_CSV,
    required=False,
    help='Path to CSV file containing train images with ids and urls.'
  )

  parser.add_argument('--path_to_validation_images_csv',
    type=str,
    default=DEFAULT_PATH_TO_VALIDATION_IMAGES_CSV,
    required=False,
    help='Path to CSV file containing validation images with ids and urls.'
  )


  args = parser.parse_args()

  setup_images_db()
  path_to_train_images_csv = args.path_to_train_images_csv
  path_to_validation_images_csv = args.path_to_validation_images_csv

  import_images_with_urls_to_db(path_to_train_images_csv, TRAIN_TABLE_NAME)
  import_images_with_urls_to_db(path_to_validation_images_csv, VALIDATION_TABLE_NAME)
