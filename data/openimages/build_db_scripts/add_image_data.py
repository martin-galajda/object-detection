import argparse
import aiosqlite
import asyncio
import aiohttp
import cv2
import numpy as np
import sys
import sqlite3
from utils.np_array_db_converters import adapt_array, convert_array
from timeit import default_timer as timer

from data.openimages.constants import Constants as OpenImagesConstants

# Converts numpy array to binary compressed version
aiosqlite.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
aiosqlite.register_converter("BLOB", convert_array)

sqlite3.register_converter("BLOB", convert_array)
sqlite3.register_adapter(np.ndarray, adapt_array)


class Constants:
    IMAGE_BYTES_COL_NAME = 'image_bytes'


async def assert_image_bytes_col_exists(db_path, table_name):
    async with aiosqlite.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
        try:
            await db_conn.executescript("""
        ALTER TABLE %s
        ADD COLUMN %s BLOB;
        """ % (table_name, Constants.IMAGE_BYTES_COL_NAME)
            )
            await db_conn.commit()

        except Exception as e:
            print(f'Error creating {Constants.IMAGE_BYTES_COL_NAME} column: {str(e)}')


async def async_get_image_from_db(image_id, db_path, table_name):
    async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
        cursor = await db_conn.execute("""
      SELECT id, url
      FROM %s
      WHERE id = ?;
    """ % table_name, (image_id,))

        db_image = await cursor.fetchone()

        return db_image


async def async_save_image_bytes_to_db(images_from_db_with_bytes, db_path, table_name):
    async with aiosqlite.connect(db_path, timeout=1000, detect_types=sqlite3.PARSE_DECLTYPES) as db_conn:
        async with db_conn.cursor() as cursor:
            await cursor.executemany("""
        UPDATE %s 
        SET image_bytes = ?
        WHERE id = ?;
      """ % table_name, images_from_db_with_bytes)

            await db_conn.commit()

            print("Saved image bytes for %d images " % int(db_conn.total_changes))


async def download_data_from_url(session, url):
    try:
        async with session.get(url) as response:
            response_text = await response.read()

            return response_text
    except Exception as exc:
        print("Error in download_data_from_url(session, url): " + str(exc), file=sys.stderr)


async def async_download_image(image_url, img_size):
    try:
        async with aiohttp.ClientSession() as session:
            bytes_img = await download_data_from_url(session, image_url)

            img_numpy = np.fromstring(bytes_img, np.uint8)

            img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

            img_resized = cv2.resize(img_cv2, (img_size, img_size))
            img = img_resized.astype(np.uint8)

            return img
    except Exception as e:
        print("Error in download_image(url, id):  " + str(e), file = sys.stderr)


async def async_get_from_db_and_download_bytes(image_idx, db_path, table_name, img_size):
    result_from_db = await async_get_image_from_db(image_idx, db_path, table_name)

    if result_from_db is None:
        return None

    image_id, img_url = result_from_db

    img_bytes = await async_download_image(img_url, img_size)
    return img_bytes, image_id


async def process_downloaded_images_with_bytes(images_from_db_with_bytes, db_path, table_name):
    await async_save_image_bytes_to_db(images_from_db_with_bytes, db_path, table_name)


async def main(args):
    current_idx = args.start_idx
    images_to_process_count = args.num_of_images_to_process
    process_images_batch_size = args.batch_process_count
    processed_count = 0
    db_path = args.path_to_images_db
    table_name = args.table_name_for_images
    max_hours_to_run = args.max_hours_to_run
    img_size = args.img_size

    max_seconds_to_run = max_hours_to_run * 60 * 60

    import_start_timestamp = timer()

    await assert_image_bytes_col_exists(db_path, table_name)
    task_list = []
    while processed_count < images_to_process_count:

        new_task = async_get_from_db_and_download_bytes(current_idx, db_path, table_name, img_size)
        if new_task is None:
            # iterated over whole dataset...
            print(f'Already processed all probably. Count: {processed_count}')
            break

        task_list += [new_task]

        current_idx += 1
        if len(task_list) % process_images_batch_size == 0:
            images_from_db_with_bytes = await asyncio.gather(*task_list)
            await process_downloaded_images_with_bytes(images_from_db_with_bytes, db_path, table_name)
            print(f'Processed up to {current_idx} images.')

            processed_count += len(task_list)
            task_list = []

        now = timer()
        if now - import_start_timestamp > max_seconds_to_run:
            print(f'Already took {now - import_start_timestamp} seconds to process, exiting...')
            break

    if len(task_list) > 0:
        images_from_db_with_bytes = await asyncio.gather(*task_list)
        await process_downloaded_images_with_bytes(images_from_db_with_bytes, db_path, table_name)
        print(f'Processed up to {current_idx} images.')

        processed_count += len(task_list)
        task_list = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_images_db',
                        type=str,
                        default=OpenImagesConstants.IMAGES_DB_PATH,
                        required=False,
                        help='Path to file containing image metadata.')

    parser.add_argument('--table_name_for_images',
                        type=str,
                        required=True,
                        help='Name of table which should be appended with image bytes. (validation_images | train_images)')

    parser.add_argument('--start_idx',
                        type=int,
                        default=1,
                        required=False,
                        help='Index (=id) of image to start processing from.')

    parser.add_argument('--num_of_images_to_process',
                        type=int,
                        default=100,
                        required=False,
                        help='Number of images to process from the table.')

    parser.add_argument('--batch_process_count',
                        type=int,
                        default=16,
                        required=False,
                        help='Number of images to process in parallel.')

    # to avoid locking databasee ...
    parser.add_argument('--max_hours_to_run',
                        type=int,
                        default=23,
                        required=False,
                        help='Maximum number of hours to perform import.')

    parser.add_argument('--img_size',
                        type=int,
                        default=225,
                        required=False,
                        help='Image width and height in pixels.')

    args = parser.parse_args()

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(main(args))