import sqlite3
import aiosqlite
import numpy as np

from data.openimages.constants import BoxableImagesConstants
from utils.np_array_db_converters import adapt_array, convert_array
from collections import defaultdict
from typing import Optional
from utils.sqlite import build_placeholder_string_from_list

# Converts numpy array to binary compressed version
aiosqlite.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
aiosqlite.register_converter("BLOB", convert_array)

sqlite3.register_converter("BLOB", convert_array)
sqlite3.register_adapter(np.ndarray, adapt_array)


def get_num_of_labels_in_db(*, db_path: str, labels_table_name: str = BoxableImagesConstants.TABLE_NAME_LABELS):
  db_image_labels_conn = sqlite3.connect(db_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute(f'''
    SELECT COUNT(*)
    FROM {labels_table_name};
  ''')

  count = cursor.fetchone()

  return count[0]

def assert_exists_db_col_index(db_path: str,  table_name: str, col_name: str):
  try:
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.executescript(f'''
      CREATE INDEX IF NOT EXISTS {table_name}_{col_name}_idx ON {table_name} ({col_name});
    ''')

    db_conn.commit()
  except Exception as e:
    print(f'Exception raised in assert_exists_db_col_index: {str(e)}')

def insert_labels_in_db(labels_data: list, *, db_path: str, labels_table_name: str = BoxableImagesConstants.TABLE_NAME_LABELS):
  db_image_labels_conn = sqlite3.connect(db_path)
  cursor = db_image_labels_conn.cursor()
  cursor.executemany(f'''
    INSERT INTO {labels_table_name} (original_label_id, label_class_name) VALUES (?, ?);
  ''', labels_data)

  cursor.close()
  print(f'Inserted {db_image_labels_conn.total_changes} labels in db: {db_path} in table: {labels_table_name}.')
  db_image_labels_conn.commit()


def sync_get_number_of_images(
    *,
    db_path: str,
    table_name_images: str
):
    db_conn = sqlite3.connect(db_path)
    db_cursor = db_conn.cursor()

    db_cursor.execute(f'''
        SELECT COUNT(*)
        FROM {table_name_images}
    ''')

    count = db_cursor.fetchone()[0]

    return count


def sync_get_boxes(
    *,
    db_path: str,
    image_ids: list,
    table_name_boxes: str,
    only_positive: bool
):
    db_conn = sqlite3.connect(db_path)
    db_cursor = db_conn.cursor()

    image_ids_placeholder = build_placeholder_string_from_list(image_ids)

    db_cursor.execute(f'''
      SELECT 
        id, 
        image_id, 
        label_id,
        x_min,
        x_max,
        y_min,
        y_max,
        confidence
      FROM {table_name_boxes}
      WHERE image_id IN ({image_ids_placeholder})
      {' AND confidence = 1.0 ' if only_positive else ''}
      ;
    ''', image_ids)

    db_boxes = db_cursor.fetchall()

    image_id_to_boxes = defaultdict(list)

    for db_box in db_boxes:
        image_id_to_boxes[db_box[1]] += [db_box]

    result = []
    for image_id in image_ids:
        result += [image_id_to_boxes[image_id]]

    image_ids_not_empty = image_id_to_boxes.keys()

    return result, image_ids_not_empty


def sync_get_images(
    *,
    db_path: str,
    image_ids: list,
    table_name_images: str,
    only_non_empty: bool
):
    db_conn = sqlite3.connect(db_path)
    db_cursor = db_conn.cursor()

    image_ids_placeholder = build_placeholder_string_from_list(image_ids)

    db_cursor.execute(f'''
      SELECT 
        id, 
        image_bytes
      FROM {table_name_images}
      WHERE image_id IN ({image_ids_placeholder})
        {' AND WHERE image_bytes IS NOT NULL ' if only_non_empty else ''}

      ;
    ''', image_ids)

    db_images = db_cursor.fetchall()

    image_id_to_images = defaultdict(list)

    for db_image in db_images:
        image_id_to_images[db_image[0]] += [db_image]

    result = []
    for image_id in image_ids:
        result += [image_id_to_images[image_id]]

    image_ids_not_empty  = image_id_to_images.keys()

    return result, image_ids_not_empty

def sync_get_non_empty_images_with_boxes(
    *,
    db_path: str,
    image_ids: list,
    table_name_images: str,
    table_name_boxes: str
):
    pass


async def async_save_boxable_images_to_db(images_data, db_path, table_name):
  async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
    async with db_conn.cursor() as cursor:
      await cursor.executemany(f'''
        INSERT INTO {table_name} (
          original_image_id,
          url,
          width,
          height,
          rotation,
          image_bytes
        ) VALUES (?,?,?,?,?,?);
      ''', images_data)

      await db_conn.commit()

      print("Saved image bytes for %d images " % int(db_conn.total_changes))


async def async_save_image_boxes_to_db(image_boxes_data, db_path, table_name):
  async with aiosqlite.connect(db_path, timeout=1000) as db_conn:
    async with db_conn.cursor() as cursor:
      await cursor.executemany(f'''
        INSERT INTO {table_name} (
          image_id,
          label_id,

          x_min,
          x_max,
          y_min,
          y_max,
          confidence,

          is_occluded,
          is_depiction,
          is_inside,
          is_truncated
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?);
      ''', image_boxes_data)

      print("Saved data for %d image boxes " % int(db_conn.total_changes))

      await db_conn.commit()


async def async_get_image_by_original_id(*, db_path: str, table_name_images: str, original_image_id: str):
  async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
    async with db_conn.cursor() as cursor:
      await cursor.execute(f'''
        SELECT id
        FROM {table_name_images}
        WHERE original_image_id = ?
      ''', (original_image_id,))

      db_img = await cursor.fetchone()
      await db_conn.commit()
      return db_img

async def async_get_label_by_original_id(*, db_path: str, table_name_labels: str, original_label_id: str):
  async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
    async with db_conn.cursor() as cursor:
      await cursor.execute(f'''
        SELECT id
        FROM {table_name_labels}
        WHERE original_label_id = ?
      ''', (original_label_id,))

      db_label = await cursor.fetchone()
      await db_conn.commit()
      return db_label

async def async_get_images_by_ids(
    *,
    db_path: str,
    table_name_images: str,
    image_ids: list,
    only_non_empty: Optional[bool] = False
):
  async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
    async with db_conn.cursor() as cursor:
      image_ids_placeholder = '?, ' * len(image_ids)
      image_ids_placeholder = image_ids_placeholder.strip()
      image_ids_placeholder = image_ids_placeholder.strip(',')
      await cursor.execute(f'''
        SELECT id, image_bytes
        FROM {table_name_images}
        WHERE id IN ({image_ids_placeholder})
            {' AND image_bytes IS NOT NULL ' if only_non_empty else '' }
        ;
      ''', image_ids)

      db_images = await cursor.fetchall()
      return db_images

async def async_get_boxes_by_image_ids(
    *,
    db_path: str,
    table_name_boxes: str,
    image_ids: list,
    only_positive: bool = False,
):
  async with aiosqlite.connect(db_path, timeout=1000) as db_conn:
    async with db_conn.cursor() as cursor:

      image_ids_placeholder = '?, ' * len(image_ids)
      image_ids_placeholder = image_ids_placeholder.strip()
      image_ids_placeholder = image_ids_placeholder.strip(',')

      await cursor.execute(f'''
        SELECT 
          id, 
          image_id, 
          label_id,
          x_min,
          x_max,
          y_min,
          y_max,
          confidence
        FROM {table_name_boxes}
        WHERE image_id IN ({image_ids_placeholder})
        {' AND confidence = 1.0 ' if only_positive else ''}
        ;
      ''', image_ids)

      db_boxes = await cursor.fetchall()

      image_id_to_boxes = defaultdict(list)

      for db_box in db_boxes:
        image_id_to_boxes[db_box[1]] += [db_box]

      result = []
      for image_id in image_ids:
        result += [image_id_to_boxes[image_id]]

      return result, list(image_id_to_boxes.keys())

async def get_non_empty_boxes(
    *,
    db_path: str,
    table_name_boxes: str,
    sampled_image_ids: list,
    required_batch: int,
    only_positive: bool
):
    results, image_ids = await async_get_boxes_by_image_ids(
        db_path=db_path,
        table_name_boxes=table_name_boxes,
        image_ids=sampled_image_ids,
        only_positive=only_positive,
    )

    print(image_ids)

    present_image_ids = []
    non_empty_results = []
    curr_idx = 0
    while len(non_empty_results) < required_batch and curr_idx < len(results):
        if len(results[curr_idx]) > 0:
            non_empty_results += [results[curr_idx]]
            present_image_ids += [image_ids[curr_idx]]
        curr_idx += 1

    return non_empty_results, present_image_ids


async def get_num_of_boxes(
    *,
    db_path: str,
    table_name_boxes: str,
):
    async with aiosqlite.connect(db_path, timeout=1000, isolation_level=None) as db_conn:
        async with db_conn.cursor() as cursor:
            await cursor.execute(f'''
                SELECT COUNT(*)
                FROM {table_name_boxes};
            ''')

            sql_results = await cursor.fetchone()
            count = sql_results[0]

            return count


async def get_num_of_images(
    *,
    db_path: str,
    table_name_images: str,
):
    async with aiosqlite.connect(db_path, timeout=1000, isolation_level=None) as db_conn:
        async with db_conn.cursor() as cursor:
            await cursor.execute(f'''
                SELECT COUNT(*)
                FROM {table_name_images};
            ''')

            sql_results = await cursor.fetchone()
            count = sql_results[0]

            return count
