from data.openimages.constants import BoxableImagesConstants
import sqlite3
import aiosqlite
from utils.np_array_db_converters import adapt_array, convert_array
import numpy as np

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
