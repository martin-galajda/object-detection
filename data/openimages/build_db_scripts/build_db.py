from data.openimages.db_utils import init, load_sql_schema_def, setup_db
import re
import os
# import tdqm
from multiprocessing import Pool
import aiosqlite
import asyncio
from time import time as timer
from itertools import islice
from time import sleep

pending_executions = 0
processed = 0


DB_FILE_PATH = './data/openimages/out/db.data'
DB_FILE_PATH = '/storage/plzen1/home/marneyko/datasets/out/db_train0.data'
# DB_FILE_PATH = '/storage/plzen1/home/marneyko/datasets/out/db.data'
# DB_FILE_PATH = '/storage/plzen1/home/marneyko/datasets/out/db.tmp.data'
SQL_SCHEMA_DEF_PATH = './data/openimages/db-schema.sql'
PATH_TO_IMAGE_METADATA_CSV = './data/openimages/train-images-with-labels-with-rotation.csv'
PATH_TO_IMAGE_URLS_TSV = './data/openimages/open-images-dataset-train0.tsv'

conn = init(DB_FILE_PATH)
cursor = conn.cursor()

sql_schema_def = load_sql_schema_def(SQL_SCHEMA_DEF_PATH)
setup_db(conn, sql_schema_def)


SQL_INSERT_IMAGE = """
  INSERT INTO images(id, url, data)
  VALUES (?, ?, NULL);
"""

SQL_GET_IMAGE_BY_URL = """
  SELECT id
  FROM images
  WHERE url = ?;
"""


async def async_batch_insert_images(sql_insert_image_update_args):
  try:
    async with aiosqlite.connect(DB_FILE_PATH, timeout = 1000.0) as db:
      try:
        await db.executemany(SQL_INSERT_IMAGE, sql_insert_image_update_args)
        await db.commit()
      except Exception as e:
        print("Exception happened " + str(e))

  except Exception as e:
    print("Exception happened " + str(e))


def batch_insert_images( sql_insert_image_update_args):
  try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_batch_insert_images(sql_insert_image_update_args))
  except Exception as e:
    print("Exception happened " + str(e))


def decrease_pending_executions(*kargs):
  global pending_executions
  global processed

  print("Decreasing pending executions")
  print(pending_executions)
  pending_executions -= 1
  print(pending_executions)


def decrease_pending_executions_on_error(*kargs):
  global pending_executions
  global processed

  print("Decreasing pending executions on error")
  pending_executions -= 1

def import_image_metadata_to_db(pool):
  global pending_executions
  global processed
  batch_insert_sql_size = 50000

  print(batch_insert_sql_size)

  logged = {}
  start = timer()
  with open(PATH_TO_IMAGE_METADATA_CSV, 'r', encoding="utf-8") as csv_images_metadata:
    csv_images_metadata.readline()

    sql_insert_image_update_args = []

    for line in csv_images_metadata:
      image_metadata_row = line.split(',')
      image_id = image_metadata_row[0]
      image_url = image_metadata_row[2]

      while pending_executions >= 1:
        print("sleeping")
        print("Pending executions: %d" % pending_executions)
        sleep(10)

      sql_insert_image_update_args += [(image_id, image_url)]

      if len(sql_insert_image_update_args) == batch_insert_sql_size:
        pending_executions += 1
        print("Adding pending %d -th execution" % pending_executions)
        pool.apply_async(batch_insert_images, (sql_insert_image_update_args,), callback = decrease_pending_executions, error_callback = decrease_pending_executions_on_error)
        sql_insert_image_update_args = []

        processed += batch_insert_sql_size

      if processed > 0 and processed % 10000 == 0 and not processed in logged:
        print("Processed %d rows with urls." % processed)
        print("Took %f seconds." % (timer() - start))
        logged[processed] = True

  while pending_executions > 0:
    sleep(10)

  pool.close()
  pool.join()

  print("Took %f seconds." % (timer() - start))

  # with open(PATH_TO_IMAGE_URLS_TSV, 'r', encoding="utf-8") as tsv_images_urls:
  #   new_tsv_file_path = re.sub('.tsv', '_enhanced.tsv', PATH_TO_IMAGE_URLS_TSV)
  #   tsv_images_urls.readline()
  #   processed = 0
  #
  #   with open(new_tsv_file_path, 'w', encoding="utf-8") as output_enhanced_image_urls_tsv:
  #     for row in tsv_images_urls:
  #       data_row = row.split('\t')
  #       image_url = data_row[0]
  #
  #       cursor.execute(SQL_GET_IMAGE_BY_URL, (image_url,))
  #       image_db_data = cursor.fetchone()
  #
  #       if image_db_data is not None:
  #         row = re.sub(os.linesep, '', row)
  #         output_enhanced_image_urls_tsv.write(row + "\t" + image_db_data[0] + os.linesep)
  #       else:
  #         print("Failed to fetch image with id for url: %s !" % image_url)
  #
  #       processed += 1
  #       if processed % 10000 == 0:
  #         print("Processed %d rows with urls." % processed)
  #
  # conn.commit()

if __name__ == '__main__':
  with Pool() as pool:  # start 4 worker processes
    import_image_metadata_to_db(pool)

