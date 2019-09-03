import argparse
import sqlite3
from data.openimages.constants import Constants
import asyncio
import aiosqlite


def setup_sql_schema(db_path, *, trainable_labels_table_name, all_labels_table_name, image_labels_table_name):
  db_conn = sqlite3.connect(db_path)
  db_cursor = db_conn.cursor()

  try:
    db_cursor.executescript(f'''
      DROP TABLE IF EXISTS {trainable_labels_table_name};
    ''')

    db_conn.commit()
  except Exception as e:
    print(f'Error when dropping SQL schema: {str(e)}')


  try:
    db_cursor.executescript(f'''
      CREATE INDEX image_labels_label_id ON image_labels (label_id);
    ''')
    db_conn.commit()
    print(f'Created image_labels_label_id INDEX on image_labels (label_id) ')
  except Exception as e:
    print(f'Error creating image_labels_label_id index: {str(e)}')

    
  try:
    db_cursor.executescript(f'''
      CREATE TABLE IF NOT EXISTS {trainable_labels_table_name} (
        id INTEGER PRIMARY KEY NOT NULL,
        original_label_id VARCHAR NOT NULL UNIQUE,
        label_class_name VARCHAR,
        label_id INT REFERENCES {all_labels_table_name}(id) NOT null,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
      );

      ALTER TABLE {all_labels_table_name}
      ADD COLUMN trainable_label_id INT REFERENCES {trainable_labels_table_name}(id);

      -- ALTER TABLE {image_labels_table_name}
      -- ADD COLUMN trainable_label_id INT REFERENCES {trainable_labels_table_name}(id);
    ''')

    db_conn.commit()
    db_conn.close()
  except Exception as e:
    print(f'Error when preparing SQL schema: {str(e)}')


  # try:
  #   db_cursor.executescript(f'''
  #       CREATE INDEX image_labels_trainable_label_id ON image_labels (trainable_label_id);
  #   ''')
  #   db_conn.commit()
  # except Exception as e:
  #   print(f'Error creating image_labels_trainable_label_id index: {str(e)}')



async def update_all_labels(db_path, all_labels_table_name, label_ids, trainable_label_ids):
  try:
    async with aiosqlite.connect(db_path, timeout=1000) as db_conn:
      db_cursor = await db_conn.cursor()

      updates_data = list(zip(trainable_label_ids, label_ids))

      await db_cursor.executemany(f'''
        UPDATE {all_labels_table_name}
        SET trainable_label_id = ?
        WHERE id = ?;
      ''', updates_data)

      await db_conn.commit()

      print("Updated %d all labels. " % int(db_conn.total_changes))
  except Exception as e:
    print(f'Error updating all_labels: {str(e)}')


async def update_image_labels(db_path, image_labels_table_name, label_ids, trainable_label_ids):

  try:
    async with aiosqlite.connect(db_path, timeout=1000) as db_conn:
      db_cursor = await db_conn.cursor()

      updates_data = list(zip(trainable_label_ids, label_ids))

      await db_cursor.executemany(f'''
        UPDATE {image_labels_table_name}
        SET trainable_label_id = ?
        WHERE label_id = ?;
      ''', updates_data)

      await db_conn.commit()

      print("Updated %d image labels. " % int(db_conn.total_changes))
  except Exception as e:
    print(f'Error updating image_labels: {str(e)}')


async def insert_trainable_label(
  db_path: str,
  *,
  all_labels_table_name: str,
  trainable_table_name: str,
  label_mid: str
):
  try:
    async with aiosqlite.connect(db_path, timeout=1000) as db_conn:
      db_cursor = await db_conn.cursor()

      await db_cursor.execute(f'''
        SELECT id, original_label_id, label_class_name
        FROM {all_labels_table_name}
        WHERE original_label_id = ?
      ''', (label_mid,))

      label_from_db = await db_cursor.fetchone()

      await db_cursor.execute(f'''
        INSERT INTO {trainable_table_name}(label_id, original_label_id, label_class_name)
        VALUES (?, ?, ?);
      ''', label_from_db)

      inserted_row_id = db_cursor.lastrowid

      await db_conn.commit()

      print("Inserted %d trainable labels. " % int(db_conn.total_changes))


    return label_from_db[0], inserted_row_id
  except Exception as e:
    print(f'Error inserting trainable_labels: {str(e)}')
    return None, None


def get_next_trainable_classes_mids(file, batch_size):
  trainable_mids = []
  eof_reached = False

  for _ in range(batch_size):
    line = file.readline()

    if line == "":
      # end of file reached
      eof_reached = True
      continue

    trainable_mids += [line.strip('\n')]

  return trainable_mids, eof_reached

async def import_trainable_label_mid(
  db_path, 
  label_mid, 
  *,
  all_labels_table_name,
  trainable_table_name,
  image_labels_table_name
):
  label_id, trainable_label_id = await insert_trainable_label(
    db_path,
    all_labels_table_name=all_labels_table_name,
    label_mid=label_mid,
    trainable_table_name=trainable_table_name
  )

  if label_id is not None and trainable_label_id is not None:
    await asyncio.gather(*[
      update_all_labels(db_path, all_labels_table_name, [label_id], [trainable_label_id]),
      # update_image_labels(db_path, image_labels_table_name, [label_id], [trainable_label_id])
    ])
  else:
    print('Skipping importing label with mid: {label_mid}')

async def main(args):
  db_path = args.path_to_image_labels_db

  trainable_labels_table_name = args.table_name_trainable_labels
  all_labels_table_name = args.table_name_all_labels
  table_name_image_labels = args.table_name_image_labels
  batch_size = args.batch_size
  start_idx = args.start_index

  if start_idx == 0:
    setup_sql_schema(
      db_path,
      trainable_labels_table_name=trainable_labels_table_name,
      all_labels_table_name=all_labels_table_name,
      image_labels_table_name=table_name_image_labels
    )

  path_to_trainable_class_file = args.path_to_trainable_classes_file

  processed_mids_count = 0
  mid_idx = 0
  with open(path_to_trainable_class_file, 'r') as trainable_classes_file:
    while True:
      task_list = []
      next_mids, eof_reached = get_next_trainable_classes_mids(trainable_classes_file, batch_size)

      for mid in next_mids:
        mid_idx += 1
        if mid_idx > start_idx:
          task_list += [import_trainable_label_mid(
            db_path,
            mid,
            all_labels_table_name=all_labels_table_name,
            trainable_table_name=trainable_labels_table_name,
            image_labels_table_name=table_name_image_labels
          )]

      await asyncio.gather(*task_list)
      processed_mids_count += batch_size

      print(f'Processed mids: {processed_mids_count}')

      if eof_reached:
        break



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--table_name_trainable_labels',
    type=str,
    default=Constants.TRAINABLE_IMAGE_LABELS_TABLE_NAME,
    required=False,
    help='Name of the table containing trainable labels.'
  )

  parser.add_argument(
    '--table_name_image_labels',
    type=str,
    default=Constants.IMAGE_LABELS_TABLE_NAME,
    required=False,
    help='Name of the table containing trainable labels.'
  )

  parser.add_argument(
    '--table_name_all_labels',
    type=str,
    default=Constants.ALL_LABELS_TABLE_NAME,
    required=False,
    help='Name of the table containing trainable labels.'
  )

  parser.add_argument(
    '--path_to_trainable_classes_file',
    type=str,
    default=Constants.PATH_TO_TRAINABLE_CLASSES_TXT_FILE,
    required=False,
    help='Path to text file containing trainable classes.'
  )

  parser.add_argument(
    '--path_to_image_labels_db',
    type=str,
    default=Constants.IMAGE_LABELS_DB_PATH,
    required=False,
    help='Path to database containing image labels.'
  )

  parser.add_argument(
    '--batch_size',
    type=int,
    default=5,
    required=False,
    help='Batch labels to import at one time.'
  )

  parser.add_argument(
    '--start_index',
    type=int,
    default=0,
    required=False,
    help='Start index to  start importing.'
  )

  args = parser.parse_args()

  event_loop = asyncio.get_event_loop()
  event_loop.run_until_complete(main(args))
