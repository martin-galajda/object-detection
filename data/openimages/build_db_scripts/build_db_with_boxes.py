from data.openimages.constants import BoxableImagesConstants
import argparse
import sqlite3


class ArgParseOptionsNames:
    path_to_db = 'path_to_db'

    train_images_table_name = 'train_images_table_name'
    train_images_boxes_table_name = 'train_images_boxes_table_name'
    train_boxes_labels_table_name = 'train_boxes_labels_table_name'

    val_images_table_name = 'val_images_table_name'
    val_images_boxes_table_name = 'val_images_boxes_table_name'
    val_boxes_labels_table_name = 'val_boxes_labels_table_name'

    labels_table_name = 'labels_table_name'


def setup_images_db(db_cursor: sqlite3.Cursor, table_name: str):
    try:
        db_cursor.executescript(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY NOT NULL,
                original_image_id VARCHAR NOT NULL UNIQUE,
                url VARCHAR NOT NULL,
                image_bytes BLOB,
                width REAL,
                height REAL,
                rotation REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                UNIQUE (url, rotation)
            )
        ''')
    except sqlite3.Error as e:
        print(f'Exception in setup_train_images_db(): {str(e)}')


# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
def setup_images_boxes_db(
    db_cursor: sqlite3.Cursor,
    table_name: str,
    referenced_image_table_name: str,
    referenced_labels_table_name: str
):
    try:
        db_cursor.executescript(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY NOT NULL,

                image_id INTEGER REFERENCES {referenced_image_table_name}(id) NOT NULL,
                label_id REFERENCES {referenced_labels_table_name}(id) NOT NULL,
        
                x_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_min REAL NOT NULL,
                y_max REAL NOT NULL,
                confidence REAL NOT NULL,
        
                is_occluded INTEGER,
                is_depiction INTEGER,
                is_inside INTEGER,
                is_truncated INTEGER,
        
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        ''')
    except sqlite3.Error as e:
        print(f'Exception in setup_images_boxes_db(): {str(e)}')


def setup_labels_db(db_cursor: sqlite3.Cursor, table_name: str):
    try:
        db_cursor.executescript(f'''
              CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY NOT NULL,
                    original_label_id VARCHAR NOT NULL UNIQUE,
                    
                    label_class_name VARCHAR NOT NULL,
                
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
              )
        ''')
    except sqlite3.Error as e:
        print(f'Exception in setup_labels_db(): {str(e)}')


def setup_sql_schema(**kwargs):
    db_conn = sqlite3.connect(kwargs[ArgParseOptionsNames.path_to_db])
    cursor = db_conn.cursor()

    # setup images schema (image id + data + metadata)
    # Setup tables for training data
    setup_images_db(cursor, table_name=kwargs[ArgParseOptionsNames.train_images_table_name])

    # Setup tables for validation data
    setup_images_db(cursor, table_name=kwargs[ArgParseOptionsNames.val_images_table_name])

    # setup labels db
    setup_labels_db(cursor, kwargs[ArgParseOptionsNames.labels_table_name])

    # setup boxes schema (reference to image + box details)
    # training data -> setup_images_boxes_db(db_cursor, table_name, referenced_image_table_name):
    setup_images_boxes_db(
        cursor,
        kwargs[ArgParseOptionsNames.train_images_boxes_table_name],
        kwargs[ArgParseOptionsNames.train_images_table_name],
        kwargs[ArgParseOptionsNames.labels_table_name],
    )
    # validation data
    setup_images_boxes_db(
        cursor,
        kwargs[ArgParseOptionsNames.val_images_boxes_table_name],
        kwargs[ArgParseOptionsNames.val_images_table_name],
        kwargs[ArgParseOptionsNames.labels_table_name],
    )


def main(args):
    kwargs = vars(args)
    setup_sql_schema(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(f'--{ArgParseOptionsNames.path_to_db}',
                        type=str,
                        default=BoxableImagesConstants.PATH_TO_DB_YOLO_V2,
                        required=False,
                        help='.')

    parser.add_argument(f'--{ArgParseOptionsNames.train_images_table_name}',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_TRAIN_BOXABLE_IMAGES,
                        required=False,
                        help='.')

    parser.add_argument(f'--{ArgParseOptionsNames.train_images_boxes_table_name}',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_TRAIN_IMAGE_BOXES,
                        required=False,
                        help='.')

    parser.add_argument(f'--{ArgParseOptionsNames.val_images_table_name}',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_VAL_BOXABLE_IMAGES,
                        required=False,
                        help='.')

    parser.add_argument(f'--{ArgParseOptionsNames.val_images_boxes_table_name}',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_VAL_IMAGE_BOXES,
                        required=False,
                        help='.')

    parser.add_argument(f'--{ArgParseOptionsNames.labels_table_name}',
                        type=str,
                        default=BoxableImagesConstants.TABLE_NAME_LABELS,
                        required=False,
                        help='.')

    main(parser.parse_args())
