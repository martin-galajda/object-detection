from data.openimages.db import DB
from data.openimages.constants import Constants
import numpy as np
from PIL import Image

def visualize_numpy_array_img(numpy_array_img, file_path = None):
    im = Image.fromarray(numpy_array_img)
    
    if file_path:
        im.save(file_path)
    else:
        im.show()


def save_candidate_placeholder(test_image_id, filepath_to_save):
    db_path = Constants.IMAGES_DB_PATH
    db = DB(db_path, Constants.IMAGE_LABELS_DB_PATH)
    images_from_db = db.get_images([test_image_id], Constants.TRAIN_TABLE_NAME_IMAGES)
    img_bytes_from_db = list(map(lambda x: x[3], images_from_db))[0]

    if img_bytes_from_db is None:
        print(f'img_bytes for image_id {test_image_id} is none')
        return
    
    visualize_numpy_array_img(img_bytes_from_db, file_path=f'{filepath_to_save}/{test_image_id}.png')

if __name__ == "__main__":
    save_candidate_placeholder(89, './resources/test_placeholders')