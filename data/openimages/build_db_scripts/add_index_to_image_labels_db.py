import sqlite3
import argparse
from data.openimages.constants import Constants


def add_indices_to_db(args):
  print(f'Adding index to image_labels table on using path {args.path_to_image_labels_db}')
  conn = sqlite3.connect(args.path_to_image_labels_db, timeout=1000)
  cursor = conn.cursor()

  try:
    cursor.executescript("""
      CREATE INDEX image_labels_original_image_id ON image_labels (original_image_id);
    """)

    conn.commit()
  except Exception as e:
    print(f'Error creating image_labels_original_image_id index: {str(e)}')

  try:
    cursor.executescript("""
      CREATE INDEX labels_trainable_label_id ON labels (trainable_label_id);
    """)

    conn.commit()
  except Exception as e:
    print(f'Error creating labels_trainable_label_id index: {str(e)}')

  conn.close()
  # print("Inserted %d image labels into db." % cursor.rowcount)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--path_to_image_labels_db',
    type=str,
    default=Constants.IMAGE_LABELS_DB_PATH,
    required=False,
    help='Path to database file containing image labels.')

  args = parser.parse_args()

  add_indices_to_db(args)
