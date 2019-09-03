import argparse
import csv
import asyncio
from data.openimages.constants import BoxableImagesConstants
from data.openimages.boxable_db import \
  get_num_of_labels_in_db, \
  insert_labels_in_db, \
  assert_exists_db_col_index, \
  async_save_image_boxes_to_db, \
  async_get_image_by_original_id, \
  async_get_label_by_original_id

class ArgParseOptionsNames:
  path_to_db = 'path_to_db'

  path_to_boxes_csv_file = 'path_to_boxes_csv_file'
  path_to_labels_csv_file = 'path_to_labels_csv_file'

  subset_type = 'subset_type'
  batch_process_size = 'batch_process_size'


# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside


def create_labels_original_label_id_index(db_path: str, labels_table_name: str):
  assert_exists_db_col_index(db_path, labels_table_name, 'original_label_id')

def create_images_original_image_id_index(db_path: str, images_table_name: str):
  assert_exists_db_col_index(db_path, images_table_name, 'original_image_id')

def import_labels(path_to_db: str, path_to_labels_csv: str, table_name: str):
  create_labels_original_label_id_index(path_to_db, table_name)

  num_of_labels_in_db = get_num_of_labels_in_db(db_path=path_to_db, labels_table_name=table_name)

  if num_of_labels_in_db > 0:
    print(f'Already  {num_of_labels_in_db} labels in {table_name}. Skipping import labels.')
    return

  print(f'num_of_labels_in_db = {num_of_labels_in_db}')

  insert_labels_in_db_data = []
  with open(path_to_labels_csv, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
      label_original_id, label_class_description = row
      insert_labels_in_db_data += [(label_original_id, label_class_description)]

  insert_labels_in_db(insert_labels_in_db_data, db_path=path_to_db, labels_table_name=table_name)

async def get_referenced_ids_for_box(*, 
  original_image_id: str,
  original_label_id: str,
  db_path: str,
  table_name_images: str,
  table_name_labels: str,
):
  jobs = [
    async_get_image_by_original_id(db_path=db_path, table_name_images=table_name_images, original_image_id=original_image_id),
    async_get_label_by_original_id(db_path=db_path, table_name_labels=table_name_labels, original_label_id=original_label_id)
  ]

  results = await asyncio.gather(*jobs)

  image_id = results[0][0] if results[0] is not None else None
  label_id = results[1][0] if results[1] is not None else None

  return image_id, label_id

async def perform_batch_insert_boxes(*,
  prepared_data_from_csv: list,
  referenced_ids_from_db: list,
  table_name_boxes: str,
  db_path: str,
):
  prepared_data_for_insert = []

  if len(prepared_data_from_csv) != len(referenced_ids_from_db):
    print(f'''
      Warning: Skipping importing box because len(prepared_data_from_csv) != len(referenced_ids_from_db) .
      "{len(prepared_data_from_csv)}" != "{len(referenced_ids_from_db)}".
    ''')
    return


  for idx, data_tuple_from_csv in enumerate(prepared_data_from_csv):
    curr_referenced_ids = referenced_ids_from_db[idx]
    referenced_image_id = curr_referenced_ids[0]
    referenced_label_id = curr_referenced_ids[1]

    if referenced_image_id is None:
      print(f'Warning: Skipping importing box because image_id for original_image_id not found.')
      continue

    if referenced_label_id is None:
      print(f'Warning: Skipping importing box because label_id for original_label_id not found.')
      continue

    prepared_data_for_insert += [(referenced_image_id, referenced_label_id, *data_tuple_from_csv)]

  try:
    await async_save_image_boxes_to_db(prepared_data_for_insert, db_path, table_name_boxes)
  except Exception as e:
    print(f'Error in async_save_image_boxes_to_db: {str(e)}')

async def import_boxes(*,
  db_path: str,
  table_name: str,
  batch_process_size: int,
  path_to_import_csv: str, 
  table_name_images: str,
  table_name_labels: str
):
  create_images_original_image_id_index(db_path, table_name_images)

  get_referenced_ids_for_boxes_jobs = []
  data_for_next_batch = []
  processed = 0
  with open(path_to_import_csv, 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:

      # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
      original_image_id = row['ImageID']
      original_label_id = row['LabelName']

      confidence = row['Confidence']
      x_min = row['XMin']
      x_max = row['XMax']

      y_min = row['YMin']
      y_max = row['YMax']

      is_occluded = row['IsOccluded']
      is_depiction = row['IsDepiction']
      is_inside = row['IsInside']
      is_truncated = row['IsTruncated']

      get_referenced_ids_for_boxes_jobs += [asyncio.ensure_future(get_referenced_ids_for_box(
        original_image_id=original_image_id,
        original_label_id=original_label_id,
        db_path=db_path,
        table_name_images=table_name_images,
        table_name_labels=table_name_labels
      ))]

      data_for_next_batch += [(
        x_min,
        x_max,
        y_min,
        y_max,
        confidence,
        is_occluded,
        is_depiction,
        is_inside,
        is_truncated
      )]

      processed += 1

      if len(get_referenced_ids_for_boxes_jobs) >= batch_process_size:
        results_referenced_ids = await asyncio.gather(*get_referenced_ids_for_boxes_jobs)
        await perform_batch_insert_boxes(
          prepared_data_from_csv=data_for_next_batch,
          referenced_ids_from_db=results_referenced_ids,
          table_name_boxes=table_name,
          db_path=db_path
        )

        print(f'Processed {processed} boxes already.')

        data_for_next_batch = []
        get_referenced_ids_for_boxes_jobs = []
      
      # Insert db order
      # image_id,
      # label_id,
      # x_min,
      # x_max,
      # y_min,
      # y_max,
      # confidence,
      # is_occluded,
      # is_depiction,
      # is_inside,
      # is_truncated
      

    if len(get_referenced_ids_for_boxes_jobs) >= 0:
      results_referenced_ids = await asyncio.gather(*get_referenced_ids_for_boxes_jobs)
      await perform_batch_insert_boxes(
        prepared_data_from_csv=data_for_next_batch,
        referenced_ids_from_db=results_referenced_ids,
        table_name_boxes=table_name,
        db_path=db_path
      )
      data_for_next_batch = []
      results_referenced_ids = []
      get_referenced_ids_for_boxes_jobs = []
    
    print(f'Processed {processed} boxes in total.')



async def main(args):
  path_to_db = args[ArgParseOptionsNames.path_to_db]
  path_to_boxes_csv_file = args[ArgParseOptionsNames.path_to_boxes_csv_file]
  subset_type = args[ArgParseOptionsNames.subset_type]

  table_name = BoxableImagesConstants.TABLE_NAME_TRAIN_IMAGE_BOXES \
    if subset_type == 'train' \
    else BoxableImagesConstants.TABLE_NAME_VAL_IMAGE_BOXES

  table_name_images = BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES \
    if subset_type == 'train' \
    else BoxableImagesConstants.TABLE_NAME_VAL_BOXABLE_IMAGES

  if not path_to_boxes_csv_file:
    path_to_boxes_csv_file = BoxableImagesConstants.PATH_TO_TRAIN_IMAGES_BOXES_CSV \
      if subset_type == 'train' \
      else BoxableImagesConstants.PATH_TO_VALIDATION_IMAGES_BOXES_CSV

  labels_descriptions_csv_path = args[ArgParseOptionsNames.path_to_labels_csv_file]

  import_labels(path_to_db, labels_descriptions_csv_path, BoxableImagesConstants.TABLE_NAME_LABELS)

  print(f'''
    Going to import boxes into
      db_path = {path_to_db},
      table_name = {table_name},
      path_to_import_csv = {path_to_boxes_csv_file},
      table_name_images = {table_name_images},
      table_name_labels = {BoxableImagesConstants.TABLE_NAME_LABELS}.
  ''')

  await import_boxes(
    db_path=path_to_db,
    table_name=table_name,
    batch_process_size = args[ArgParseOptionsNames.batch_process_size],
    path_to_import_csv = path_to_boxes_csv_file, 
    table_name_images=table_name_images,
    table_name_labels=BoxableImagesConstants.TABLE_NAME_LABELS
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(f'--{ArgParseOptionsNames.path_to_db}', 
                      type=str,
                      default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                      required=False,
                      help='Path to db.')

  parser.add_argument(f'--{ArgParseOptionsNames.batch_process_size}', 
                      type=int,
                      default=50,
                      required=False,
                      help='Batch process size. Default 50.')

  parser.add_argument(f'--{ArgParseOptionsNames.subset_type}', 
                      type=str,
                      required=True,
                      choices=['train', 'validation'],
                      help='Subset type (train | validation).')

  parser.add_argument(f'--{ArgParseOptionsNames.path_to_boxes_csv_file}', 
                      type=str,
                      default=None,
                      required=False,
                      help='Path to CSV containing image boxes data.')

  parser.add_argument(f'--{ArgParseOptionsNames.path_to_labels_csv_file}', 
                      type=str,
                      default=BoxableImagesConstants.PATH_TO_BOXABLE_LABELS_DESCRIPTIONS_CSV,
                      required=False,
                      help='Path to CSV containing boxable labels data.')

  args = vars(parser.parse_args())

  event_loop = asyncio.get_event_loop()
  event_loop.run_until_complete(main(args))
