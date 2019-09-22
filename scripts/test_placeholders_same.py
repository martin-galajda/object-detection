from data.openimages.constants import Constants
from data.openimages.db import DB
import numpy as np

def get_placeholder_as_numpy_array():
    restored_placeholder_np_array = np.load(f'{Constants.PLACEHOLDERS_PATH}/placeholder_img_225_2.npy')
    
    return restored_placeholder_np_array

def test_placeholders_equal(test_image_id):
    placeholder_np_array = get_placeholder_as_numpy_array()
    
    db = DB(Constants.IMAGES_DB_PATH, Constants.IMAGE_LABELS_DB_PATH)
    images_from_db = db.get_images([test_image_id], Constants.TRAIN_TABLE_NAME_IMAGES)
    img_bytes_from_db = list(map(lambda x: x[3], images_from_db))[0]
    
    print(f'np_array_equal = {np.array_equal(placeholder_np_array, img_bytes_from_db)}')


    np.save(f'{Constants.PLACEHOLDERS_PATH}/test', img_bytes_from_db)
    reloaded_from_file = np.load(f'{Constants.PLACEHOLDERS_PATH}/test.npy')
    print(f'np_array_equal = {np.array_equal(img_bytes_from_db, reloaded_from_file)}')



if __name__ == "__main__":
    test_placeholders_equal(89)
