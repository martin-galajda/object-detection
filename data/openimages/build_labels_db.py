import csv
import sqlite3
from collections import namedtuple
import argparse

csv_file_with_image_labels = './data/openimages/train-annotations-human-imagelabels.csv'
csv_file_with_class_descriptions = './data/openimages/class-descriptions.csv'

DB_PATH = './data/openimages/out/db.labels.data'

Label = namedtuple('Label', 'id, original_label_id, label_class_name, created_at, updated_at')
Image = namedtuple('Image', 'id, original_image_id, label_id, label_source, confidence, created_at, updated_at')

def setup_labels_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  cursor.executescript("""
    CREATE TABLE IF NOT EXISTS labels (
      id INTEGER PRIMARY KEY NOT NULL,
      original_label_id VARCHAR NOT NULL UNIQUE,
      label_class_name VARCHAR,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
    );

    CREATE TABLE  IF NOT EXISTS image_labels (
      id INTEGER PRIMARY KEY NOT NULL,
      original_image_id VARCHAR NOT NULL,
      label_id INT REFERENCES labels(id) NOT null,
      label_source VARCHAR NOT NULL,
      confidence REAL NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
    );

  """)

  conn.commit()

def insert_labels_into_db(labels):
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  cursor.executemany("""
    INSERT into labels (original_label_id, label_class_name)
    VALUES (?, ?);
  """, labels)

  print("Inserted %d labels into db." % cursor.rowcount)

  conn.commit()

def insert_image_labels_into_db(image_labels):
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  cursor.executemany("""
    INSERT INTO image_labels (original_image_id, label_id, label_source, confidence)
    VALUES (?, ?, ?, ?);
  """, image_labels)

  # print("Inserted %d image labels into db." % cursor.rowcount)

  conn.commit()


def get_labels_by_original_ids(original_ids):
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()

  image_ids_placeholders = ','.join(['?' for _ in range(len(original_ids))])

  cursor.execute("""
    SELECT id, original_label_id, label_class_name, created_at, updated_at
    FROM labels
    WHERE original_label_id IN (%s);
  """ % image_ids_placeholders, original_ids)

  values = list(map(Label._make, cursor.fetchall()))


  return values

def import_label_set_to_db():
  batch_update_size = 200
  labels = []

  processed = 0
  with open(csv_file_with_class_descriptions, 'r', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      labels += [(row[0], row[1])]

      if len(labels) > 0 and ((len(labels) % batch_update_size) == 0):
        insert_labels_into_db(labels)

        processed += len(labels)
        labels = []

  if len(labels) > 0:
    insert_labels_into_db(labels)
    processed += len(labels)

  print("Processed %d labels." % processed)


def save_image_labels(label_images):
  label_original_ids = list(map(lambda label_image: label_image[1], label_images))

  labels_from_db = get_labels_by_original_ids(label_original_ids)

  original_id_to_label_from_db = {}

  for label_from_db in labels_from_db:
    original_id_to_label_from_db[label_from_db.original_label_id] = label_from_db


  labels_for_db_update = []
  for label in label_images:
    label_id = original_id_to_label_from_db[label[1]].id
    labels_for_db_update += [(label[0], label_id, label[2], label[3])]

  insert_image_labels_into_db(labels_for_db_update)


def import_image_labels_to_db(path_to_image_labels_file):
  batch_update_size = 5000
  label_images = []

  processed = 0

  with open(path_to_image_labels_file, 'r', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if not row['ImageID'] or not row['LabelName']:
        continue
      label_images += [(row['ImageID'], row['LabelName'], row['Source'], row['Confidence'])]

      if len(label_images) > 0 and ((len(label_images) % batch_update_size) == 0):
        save_image_labels(label_images)
        processed += len(label_images)
        label_images = []

      if len(label_images) == 0 and processed > 0 and ((processed % 50000) == 0):
        print("Processed %d image labels." % processed)

  if len(label_images) > 0:
    save_image_labels(label_images)
    processed += len(label_images)

  print("Processed %d image labels." % processed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--table', type=str, default='labels', required=False,
              help='Table to build (labels|image_labels).')

  parser.add_argument('--path_to_image_labels_file',
                      type=str,
                      default=csv_file_with_image_labels,
                      required=False,
                      help='Path to file containing image labels.')

  args = parser.parse_args()

  setup_labels_db()
  if args.table == 'labels':
    import_label_set_to_db()
  elif args.table == 'image_labels':
    path_to_image_labels_file = args.path_to_image_labels_file
    import_image_labels_to_db(path_to_image_labels_file)
