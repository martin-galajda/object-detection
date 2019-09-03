from data.openimages.constants import BoxableImagesConstants
import argparse
import csv
import asyncio
import aiosqlite
from data.openimages.import_db_scripts.import_db_utils import async_download_image
from data.openimages.boxable_db import async_save_boxable_images_to_db
from timeit import default_timer as timer
from common.argparse_types import str2bool
from typing import Dict


class ArgParseOptionsNames:
    path_to_db = 'path_to_db'

    num_of_images = 'num_of_images'
    start_idx = 'start_idx'

    batch_process_size = 'batch_process_size'
    subset_type = 'subset_type'

    target_img_width = 'target_img_width'
    target_img_height = 'target_img_height'

    max_hours_to_run = 'max_hours_to_run'
    max_running_db_jobs = 'max_running_db_jobs'

    letterbox_image = 'letterbox_image'


class SubsetTypes:
    validation = 'validation'
    train = 'train'


async def get_last_imported_image(db_path, image_table_name):
    async with aiosqlite.connect(db_path) as db_conn:
        cursor = await db_conn.cursor()

        await cursor.execute(f'''
            SELECT id, original_image_id
            FROM {image_table_name}
            ORDER BY id DESC
            LIMIT 1;
        ''')

        last_img = await cursor.fetchone()

        return last_img


async def download_img(
    original_image_id,
    thumbnail_url,
    target_img_width,
    target_img_height,
    rotation,
    original_url,
    letterbox_image
):
    url_downloaded = None
    try:
        image_bytes = await async_download_image(
            thumbnail_url,
            (target_img_width, target_img_height),
            letterbox_image=letterbox_image
        )
        url_downloaded = thumbnail_url
    except Exception as e:
        print(f'Exception occured in import_img().'
              f'Failed to download thumbnail url: {str(e)}... trying original url')
        image_bytes = None

    if image_bytes is None:
        try:
            img_dimensions = (target_img_width, target_img_height)
            image_bytes = await async_download_image(
                original_url,
                img_dimensions,
                letterbox_image=letterbox_image
            )
            url_downloaded = original_url
        except Exception as e:
            print(f'Exception occured in import_img().'
                  f'Failed to download using original url: {str(e)}... Failing')
            return None

    return (
        original_image_id,
        url_downloaded,
        target_img_width,
        target_img_height,
        rotation,
        image_bytes
    )


async def wait_until_all_tasks_done(task_list):
    some_task_not_done = True
    while some_task_not_done:
        await asyncio.sleep(1)
        some_task_not_done = False

        for task in task_list:
            if not task.done():
                some_task_not_done = True


async def main(arguments: Dict[str, any]):
    max_seconds_to_run = arguments[ArgParseOptionsNames.max_hours_to_run] * 60 * 60
    import_start_timestamp = timer()

    db_path = arguments[ArgParseOptionsNames.path_to_db]
    batch_process_size = arguments[ArgParseOptionsNames.batch_process_size]

    target_img_width = arguments[ArgParseOptionsNames.target_img_width]
    target_img_height = arguments[ArgParseOptionsNames.target_img_height]
    subset_type = arguments[ArgParseOptionsNames.subset_type]
    max_number_of_images_to_import = arguments[ArgParseOptionsNames.num_of_images] \
        if ArgParseOptionsNames.num_of_images in arguments \
        else None

    max_running_db_jobs = arguments[ArgParseOptionsNames.max_running_db_jobs]
    csv_file_with_images = BoxableImagesConstants.PATH_TO_TRAIN_IMAGES_CSV \
        if subset_type == SubsetTypes.train \
        else BoxableImagesConstants.PATH_TO_VALIDATION_IMAGES_CSV

    table_name_for_import = BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES \
        if subset_type == SubsetTypes.train \
        else BoxableImagesConstants.TABLE_NAME_VAL_BOXABLE_IMAGES
    letterbox_image = arguments[ArgParseOptionsNames.letterbox_image]

    last_img = await get_last_imported_image(db_path, table_name_for_import)
    last_img_already_found = last_img is None

    print(f'last_img_already_found: {last_img_already_found}')

    processed = 0

    task_list_downloads = []
    task_list_db_updates = []
    with open(csv_file_with_images, 'r', encoding='utf8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            now = timer()
            if now - import_start_timestamp > max_seconds_to_run:
                print(f'Already took {now - import_start_timestamp} seconds to process, going to finish...')
                break

            original_image_id = row['ImageID'].strip()

            if not last_img_already_found:
                if last_img[1] == original_image_id:
                    last_img_already_found = True
                    print(f'Last image imported found ... starting to download')
                processed += 1
                continue

            if max_number_of_images_to_import is not None and processed >= max_number_of_images_to_import:
                print(f'Already processed {max_number_of_images_to_import} images, going to finish...')
                break

            # ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation
            rotation = row['Rotation']
            thumbnail_url = row['Thumbnail300KURL']
            original_url = row['OriginalURL']

            task_list_downloads += [download_img(
                original_image_id,
                thumbnail_url,
                target_img_width,
                target_img_height,
                rotation,
                original_url,
                letterbox_image
            )]

            if len(task_list_downloads) % batch_process_size == 0:
                downloaded_images_data = await asyncio.gather(*task_list_downloads)
                print(downloaded_images_data)
                downloaded_images_data = filter(lambda x: x is not None and x[5] is not None, downloaded_images_data)
                task_list_downloads = []
                task_list_db_updates += [asyncio.ensure_future(
                    async_save_boxable_images_to_db(downloaded_images_data, db_path, table_name_for_import))
                ]
                processed += batch_process_size

            if len(task_list_db_updates) >= max_running_db_jobs:
                await wait_until_all_tasks_done(task_list_db_updates)
                task_list_db_updates = []

                print(f'Processed in total images: {processed}.')

    if len(task_list_downloads) > 0:
        downloaded_images_data = await asyncio.gather(*task_list_downloads)
        task_list_downloads = []
        downloaded_images_data = filter(lambda x: x is not None and x[5] is not None, downloaded_images_data)
        await async_save_boxable_images_to_db(downloaded_images_data, db_path, table_name_for_import)

    if len(task_list_db_updates) > 0:
        await asyncio.gather(*task_list_db_updates)
        task_list_db_updates = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(f'--{ArgParseOptionsNames.path_to_db}',
                        type=str,
                        default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                        required=False,
                        help='Path to db.')

    parser.add_argument(f'--{ArgParseOptionsNames.target_img_height}',
                        type=int,
                        default=448,
                        required=False,
                        help='Target image height.')

    parser.add_argument(f'--{ArgParseOptionsNames.target_img_width}',
                        type=int,
                        default=448,
                        required=False,
                        help='Target image width.')

    parser.add_argument(f'--{ArgParseOptionsNames.batch_process_size}',
                        type=int,
                        default=50,
                        required=False,
                        help='Batch process size. Default 50.')

    parser.add_argument(f'--{ArgParseOptionsNames.subset_type}',
                        type=str,
                        required=True,
                        choices=[SubsetTypes.train, SubsetTypes.validation],
                        help=f'Subset type ({SubsetTypes.train} | {SubsetTypes.validation}).')

    parser.add_argument(f'--{ArgParseOptionsNames.max_hours_to_run}',
                        type=int,
                        default=23,
                        required=False,
                        help='Maximum numbers of hours to run for the script.')

    parser.add_argument(f'--{ArgParseOptionsNames.max_running_db_jobs}',
                        type=int,
                        default=10,
                        required=False,
                        help='Maximum numbers of db jobs to run for the script.')

    parser.add_argument(f'--{ArgParseOptionsNames.num_of_images}',
                        type=int,
                        required=False,
                        default=None,
                        help='Maximum number of images to import.')

    parser.add_argument(f'--{ArgParseOptionsNames.letterbox_image}',
                        type=str2bool,
                        required=False,
                        default=False,
                        help='Flag to determine whether letterbox image before saving.')

    args = vars(parser.parse_args())

    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(main(args))
