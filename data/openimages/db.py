import sqlite3
from data.openimages.constants import Constants as OpenImageDataConstants

def get_positive_image_labels_from(original_image_ids, *, db_image_labels_path = OpenImageDataConstants.IMAGES_DB_PATH):
  original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

  db_image_labels_conn = sqlite3.connect(db_image_labels_path, timeout=1000)
  cursor = db_image_labels_conn.cursor()
  cursor.execute("""
    SELECT id, original_image_id, label_id
    FROM image_labels
    WHERE original_image_id IN (%s) AND confidence = 1.0;
  """ % original_image_ids_placeholder, original_image_ids)

  return cursor.fetchall()

def get_image_labels_by_original_image_ids(original_image_ids, *, db_labels_path = OpenImageDataConstants.IMAGE_LABELS_DB_PATH):
  original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

  db_image_labels_conn = sqlite3.connect(db_labels_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute("""
    SELECT image_labels.id, image_labels.original_image_id, image_labels.label_id, labels.label_class_name
    FROM image_labels
    INNER JOIN labels ON image_labels.label_id = labels.id
    WHERE original_image_id IN (%s) AND confidence > 0.0;
  """ % original_image_ids_placeholder, original_image_ids)

  labels = cursor.fetchall()

  return labels



def get_labels_by_ids(ids, *, db_labels_path = OpenImageDataConstants.IMAGE_LABELS_DB_PATH):
  ids_placeholder = ','.join(['?' for _ in range(len(ids))])
  db_image_labels_conn = sqlite3.connect(db_labels_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute("""
    SELECT id, label_class_name
    FROM labels
    WHERE id IN (%s)
  """ % ids_placeholder, ids)

  labels = cursor.fetchall()

  return labels

def get_trainable_label_by_ids(ids, *, db_labels_path = OpenImageDataConstants.IMAGE_LABELS_DB_PATH):
  ids_placeholder = ','.join(['?' for _ in range(len(ids))])
  db_image_labels_conn = sqlite3.connect(db_labels_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute(f'''
    SELECT id, label_class_name
    FROM {OpenImageDataConstants.TRAINABLE_IMAGE_LABELS_TABLE_NAME}
    WHERE id IN ({ids_placeholder})
  ''', ids)

  labels = cursor.fetchall()

  return labels


def get_trainable_labels_by_original_image_ids(original_image_ids, *, db_labels_path = OpenImageDataConstants.IMAGE_LABELS_DB_PATH):
  original_image_ids_placeholder = ','.join(['?' for _ in range(len(original_image_ids))])

  db_image_labels_conn = sqlite3.connect(db_labels_path)
  cursor = db_image_labels_conn.cursor()
  cursor.execute(f'''
    SELECT
      {OpenImageDataConstants.IMAGE_LABELS_TABLE_NAME}.id,
      {OpenImageDataConstants.IMAGE_LABELS_TABLE_NAME}.original_image_id,
      {OpenImageDataConstants.IMAGE_LABELS_TABLE_NAME}.label_id,
      {OpenImageDataConstants.ALL_LABELS_TABLE_NAME}.label_class_name
    FROM {OpenImageDataConstants.IMAGE_LABELS_TABLE_NAME}
    INNER JOIN {OpenImageDataConstants.ALL_LABELS_TABLE_NAME} 
      ON {OpenImageDataConstants.IMAGE_LABELS_TABLE_NAME}.label_id = {OpenImageDataConstants.ALL_LABELS_TABLE_NAME}.id
    INNER JOIN {OpenImageDataConstants.TRAINABLE_IMAGE_LABELS_TABLE_NAME} 
      ON 
        {OpenImageDataConstants.TRAINABLE_IMAGE_LABELS_TABLE_NAME}.id = {OpenImageDataConstants.ALL_LABELS_TABLE_NAME}.trainable_label_id
    WHERE 
      original_image_id IN ({original_image_ids_placeholder})
      AND {OpenImageDataConstants.ALL_LABELS_TABLE_NAME}.trainable_label_id IS NOT NULL;
    ;
  ''', original_image_ids)

  labels = cursor.fetchall()

  return labels
