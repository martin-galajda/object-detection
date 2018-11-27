import argparse
import re
import os
import io
import requests
# from PIL import Image
from urllib.request import urlretrieve
import numpy as np
# import matplotlib.pyplot as plt
from time import time as timer
from io import BytesIO
import asyncio
import aiohttp
import aiosqlite
import cv2
from multiprocessing import Pool, Queue
from time import sleep
from itertools import islice
import sqlite3
import  gc
import zlib
import codecs


# DB_FILE_PATH = './data/openimages/out/db2.data'
DB_FILE_PATH = '/storage/plzen1/home/marneyko/datasets/out/db.data'

processed = 0

class Enums:
  NOTIFICATION_TYPE_DOWNLOADED = 0
  NOTIFICATION_TYPE_SAVED_TO_DB = 1

def adapt_array(arr):
  """
  http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
  """

  out = io.BytesIO()
  out.seek(0)
  np.savez_compressed(out, np.array(arr))
  out.seek(0)

  return out.read()
 
 # return buffer(list(arr))


def convert_array(text):
  out = io.BytesIO(text)
  out.seek(0)
  return np.load(out)['arr_0']

 # return np.array(list(text)).reshape((224, 224, 3))

# Converts np.array to TEXT when inserting
aiosqlite.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
aiosqlite.register_converter("BLOB", convert_array)

async def save_image_to_db(image_id, image_pixels_array):
  try:
    async with aiosqlite.connect(DB_FILE_PATH, timeout = 1000, detect_types=sqlite3.PARSE_DECLTYPES) as db:
      await db.execute("""
        UPDATE images
        SET image_data = ?
        WHERE id = ?;
      """, (image_pixels_array, image_id))

      await db.commit()

      print("Saved to db successfully...")
  except Exception as e:
    print("Exception occured when saving image to db: %s " % str(e))

async def download_data_from_url(session, url):
  try:
    async with session.get(url) as response:
      response_text = await response.read()

      return response_text
  except Exception as exc:
    print(exc)

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
    print("Error downloading image " + str(e))


def build_bulk_update_image_data_sql(updates):
  update_sql_str = """
    UPDATE images
    SET image_data = CASE id
  """

  sql_query_values = []
  for  update in updates:
    update_sql_str += """
      WHEN ? THEN ?
    """
    sql_query_values += [update[1], update[0]]
  update_sql_str += "END "

  ids_placeholders_str = ','.join(['?' for _ in updates])
  update_sql_str += "WHERE id in (" + ids_placeholders_str + ");"

  sql_query_values += [str(update[1]) for update in updates]

  return update_sql_str, sql_query_values
async def save_images(sql_values_for_db_update):
  start = timer()
  print("Preparing to save %d values to db..." % len(sql_values_for_db_update))

  try:
    async with aiosqlite.connect(DB_FILE_PATH, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db:
      print("Saving %d values to db..." % len(sql_values_for_db_update))

      await db.executemany("""
          UPDATE images
          SET image_data = ?
          WHERE id = ?;
      """, sql_values_for_db_update)
      # sql, values = build_bulk_update_image_data_sql(sql_values_for_db_update)
      # cursor = await db.cursor()
      # await cursor.execute(sql, values)

      global_queue.put([Enums.NOTIFICATION_TYPE_SAVED_TO_DB, len(sql_values_for_db_update), timer(), timer() - start])

      # print("Saved %s values to db successfully..." % str(cursor.rowcount))

      # await cursor.execute("""
      #   SELECT id, image_data
      #   FROM images
      #   WHERE id = ?;
      # """, (values[len(values) - 1],))
      # image_from_db = await cursor.fetchone()

      # print("Image data from db reconstructed: ")
      # print(image_from_db[1])

      await db.commit()

      

  except Exception as e:
    print("Exception occured when saving image to db: %s " % str(e))


def save_images_worker(sql_values_for_db_update):
  try:
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    loop = asyncio.get_event_loop()

    sql_values_for_db_update = list(filter(lambda value: value is not None and value[0] is not None and value[1] is not None, sql_values_for_db_update))

    loop.run_until_complete(asyncio.gather(save_images(sql_values_for_db_update)))

    print("Finishing save_images_worker")
  except Exception as e:  
    print("Exception in save_images_worker occured : %s!" % str(e))


async def async_download_images(lines):
  try:
    futures = []
    download_start = timer()
    for line in lines:
      cols = line.split('\t')
      image_url, image_id = cols

      image_id = image_id.strip(os.linesep + " ")

      futures += [download_image(image_url, image_id)]

    # print("Downloading and saving %d images." % len(futures))
    sql_values_for_db_update = await asyncio.gather(*futures)

    global_queue.put([Enums.NOTIFICATION_TYPE_DOWNLOADED, len(futures), timer() - download_start])

    # print("Downloaded %d images in process with %d/" % (len(futures), os.getpid()))

    # await asyncio.wait([save_images(sql_values_for_db_update)])

    # global_queue.put([Enums.NOTIFICATION_TYPE_SAVED_TO_DB, len(lines), timer()])

    # print("Downloaded and saved %d images." % len(futures))

    return sql_values_for_db_update

  except Exception as e:
    print("Error downloading and saving images " + str(e))


def download_images_from_lines(lines):
  try:
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    loop = asyncio.get_event_loop()
    sql_values_for_db_update = loop.run_until_complete(asyncio.gather(async_download_images(lines)))

    print("Finishing download_images_from_lines")

    return sql_values_for_db_update
  except Exception as e:  
    print("Exception occured : %s!" % str(e))

def download_and_save_images(pool, shared_queue, image_urls_file_path):
  enqueued = 0
  batch_download_size = 100
  lines_processed = 0

  start = timer()
  jobs_to_be_secheduled = []
  max_concurrent_jobs = 5

  worker_tasks = []
  db_update_tasks = []

  print("Using %d processes." % pool._processes)

  with open(image_urls_file_path) as tsv_file:
    lines = []
    slice = islice(tsv_file, 0, 2000)

    for line in slice:

      lines += [line]

      if len(lines) > 0 and len(lines) % batch_download_size == 0:
        enqueued += len(lines)

        jobs_to_be_secheduled += [lines]
        lines = []

        worker_tasks += [pool.map_async(download_images_from_lines, (lines,))]

      if len(worker_tasks) == max_concurrent_jobs:
        # worker_tasks = [pool.apply_async(download_and_save_images_from_lines, (data_for_job, )) for data_for_job in jobs_to_be_secheduled]
        flattened_sql_values_for_db_update = []
        for worker_task in worker_tasks:
          sql_values_for_db_update = worker_task.get()[0][0]
          flattened_sql_values_for_db_update += sql_values_for_db_update

        worker_tasks = []
        db_update_tasks += [pool.map_async(save_images_worker, (flattened_sql_values_for_db_update,))]

        while not shared_queue.empty():
          queue_data = shared_queue.get()
          notif_type = queue_data[0]

          if notif_type == Enums.NOTIFICATION_TYPE_DOWNLOADED:
            _, lines_downloaded, time_to_download = queue_data
            print("Downloaded %d lines in %f seconds." % (lines_downloaded, time_to_download))
          elif notif_type == Enums.NOTIFICATION_TYPE_SAVED_TO_DB:
            _, worker_lines_processed, time_finished, time_saving_db = queue_data
            lines_processed += worker_lines_processed
            print("Processed %d lines in %f seconds." % (lines_processed, time_finished - start))
            print("Processed %f seconds to save into db." % (time_saving_db,))

        jobs_to_be_secheduled = []

      if len(db_update_tasks) > 0:
        print("Awaiting tasks for sql db updates")
        [db_update_task.wait() for db_update_task in db_update_tasks]
        db_update_tasks = []

    if len(db_update_tasks) > 0:
      print("Awaiting tasks for sql db updates")
      [db_update_task.wait() for db_update_task in db_update_tasks]
      db_update_tasks = []

  pool.close()
  pool.join()

  print("Took %f seconds" % (timer() - start))

def init_worker(_queue):
  ''' store the queue for later use '''
  global global_queue
  global_queue = _queue


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_train_data', dest = 'path_to_train_data_dest', default = './data/openimages/open-images-dataset-train0_enhanced.tsv')
  #parser.add_argument('--path_to_train_data', dest = 'path_to_train_data_dest', default = './data/openimages/open-images-train0_enhanced_first_2000.tsv')
  #parser.add_argument('--path_to_train_data', dest='path_to_train_data_dest', default='./data/openimages/sample.tsv')
  # parser.add_argument('--out_train_data', dest = 'out_train_data_dest', default = '/storage/brno7-cerit/home/marneyko/datasets/out/train_images0.csv')
  # parser.add_argument('--out_train_data', dest = 'out_train_data_dest', default = '/storage/plzen1/home/marneyko/datasets/out/train_images0_v2.csv')
  # parser.add_argument('--out_train_data', dest='out_train_data_dest', default='./data/openimages/out/train_images0_v2_enhanced.csv')
  # parser.add_argument('--out_train_data', dest = 'out_train_data_dest', default = '/storage/brno7-cerit/home/marneyko/datasets/out/testing.csv')

  args, unknown = parser.parse_known_args()

  file_train_image_urls = args.path_to_train_data_dest

  queue = Queue()
  with Pool(initializer = init_worker, initargs = (queue, )) as pool:  # start 4 worker processes
    download_and_save_images(pool, queue, file_train_image_urls)


