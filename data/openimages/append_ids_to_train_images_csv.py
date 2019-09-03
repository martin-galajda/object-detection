import aiosqlite
import asyncio
from time import time as timer
from itertools import islice
import re
import os
from data.openimages.constants import Constants

pending_executions = 0
processed = 0

DB_FILE_PATH = './data/openimages/out/db.data'
PATH_TO_IMAGE_URLS_TSV = f'{Constants.METADATA_FILES_DIR_PATH}/open-images-dataset-train0.tsv'

process_rows_batch_size = 10000

SQL_GET_IMAGE_BY_URL = """
  SELECT id, url
  FROM images
  WHERE url IN (%s)
  ORDER BY url;
"""

placeholders = ['?' for _ in range(process_rows_batch_size)]
placeholders = ', '.join(placeholders)

SQL_GET_IMAGE_BY_URL = SQL_GET_IMAGE_BY_URL % placeholders


def build_enhanced_csv_row(row):
  return row[1] + '\t' + row[0] + os.linesep


async def async_batch_update_csv(tsv_file, rows_data):
  try:
    async with aiosqlite.connect(DB_FILE_PATH, timeout = 1000.0) as db:
      cursor = await db.execute(SQL_GET_IMAGE_BY_URL, [row[0] for row in rows_data])
      rows = await cursor.fetchall()

      if len(rows_data) != len(rows):
        print("Fetched %d rows but expected %d" % (len(rows), len(rows_data)))
        results = [tsv_file.write(build_enhanced_csv_row(row)) for row in rows]
        print("Written %d lines." % len(rows))
      else:
        results = [tsv_file.write(build_enhanced_csv_row(row)) for row in rows]
        print("Written %d lines." % len(rows))
  except Exception as e:
    print("Exception happened " + str(e))

async def create_index_on_urls():
  async with aiosqlite.connect(DB_FILE_PATH, timeout=1000.0) as db:
    await db.executescript("""
      CREATE INDEX url_index ON images (url);
    """)

    await db.commit()


async def append_ids_to_csv():
  global process_rows_batch_size
  logged = {}
  with open(PATH_TO_IMAGE_URLS_TSV, 'r', encoding="utf-8") as tsv_images_urls:
    new_tsv_file_path = re.sub('.tsv', '_enhanced.tsv', PATH_TO_IMAGE_URLS_TSV)
    tsv_images_urls.readline()
    with open(new_tsv_file_path, 'w', encoding="utf-8") as output_enhanced_image_urls_tsv:
      processed = 0

      rows_data = []

      for row in tsv_images_urls:
        row_data = row.split('\t')

        if len(row_data) == 3:
          rows_data += [row_data]
        else:
          print("Invalid line %s" % row)


        if len(rows_data) > 0 and len(rows_data) % process_rows_batch_size == 0:
          await async_batch_update_csv(output_enhanced_image_urls_tsv, rows_data)
          processed += len(rows_data)
          rows_data = []

        if processed > 0 and processed % 100000 == 0 and not processed in logged:
          print("Processed %d rows." % processed)
          logged[processed] = True

      if len(rows_data) > 0:
        await async_batch_update_csv(output_enhanced_image_urls_tsv, rows_data)
        rows_data = []


if __name__ == '__main__':
  loop = asyncio.get_event_loop()
  loop.run_until_complete(append_ids_to_csv())
