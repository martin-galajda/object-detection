
class Constants:
  NUM_OF_CLASSES = 19995
  NUM_OF_TRAINABLE_CLASSES = 7186

  METADATA_FILES_DIR_PATH = './tsv_files'

  IMAGE_LABELS_DB_PATH = './db/db.labels.data'
  IMAGES_DB_PATH = './db/db.images.data'

  FILE_PATH_TO_TRAIN_IMAGE_URLS = f'{METADATA_FILES_DIR_PATH}/open-images-dataset-train0_enhanced.tsv'

  PATH_TO_TRAINABLE_CLASSES_TXT_FILE = f'{METADATA_FILES_DIR_PATH}/classes-trainable.txt'

  TRAINABLE_IMAGE_LABELS_TABLE_NAME = 'trainable_labels'
  ALL_LABELS_TABLE_NAME = 'labels'
  IMAGE_LABELS_TABLE_NAME = 'image_labels'

  TRAIN_TABLE_NAME_IMAGES = 'train_images'
  VALIDATION_TABLE_NAME_IMAGES = 'validation_images'

  TOTAL_NUMBER_OF_SAMPLES = 5000000

  DATETIME_FORMAT = '%Y-%m-%d_%H:%M:%S'

class BoxableImagesConstants:
  PATH_TO_DB_YOLO_V2 = './db/boxable-images-yolo-v2.data'

  NUM_OF_CLASSES = 601

  ### Constants related to boxable subset
  TABLE_NAME_TRAIN_BOXABLE_IMAGES = 'train_boxable_images'
  TABLE_NAME_VAL_BOXABLE_IMAGES = 'val_boxable_images'

  TABLE_NAME_TRAIN_IMAGE_BOXES = 'train_images_boxes'
  TABLE_NAME_VAL_IMAGE_BOXES = 'val_images_boxes'

  TABLE_NAME_LABELS = 'labels'

  PATH_TO_VALIDATION_IMAGES_CSV = f'{Constants.METADATA_FILES_DIR_PATH}/boxable/validation-images-with-rotation.csv'
  PATH_TO_TRAIN_IMAGES_CSV = f'{Constants.METADATA_FILES_DIR_PATH}/boxable/train-images-boxable-with-rotation.csv'

  PATH_TO_VALIDATION_IMAGES_BOXES_CSV = f'{Constants.METADATA_FILES_DIR_PATH}/boxable/validation-annotations-bbox.csv'
  PATH_TO_TRAIN_IMAGES_BOXES_CSV = f'{Constants.METADATA_FILES_DIR_PATH}/boxable/train-annotations-bbox.csv'
  
  ATH_TO_BOXABLE_LABELS_DESCRIPTIONS_CSV = f'{Constants.METADATA_FILES_DIR_PATH}/boxable/class-descriptions-boxable.csv'

