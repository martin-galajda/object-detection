import cv2
import numpy as np


def resize_and_letter_box(image, *, target_width: int, target_height: int, interpolation=cv2.INTER_LINEAR):
    """
    Resize and letter box image to have a desired width and height.
    Image is resized with cv2.INTER_LINEAR interpolation strategy.

    :param image: numpy.ndarray((image_rows, image_cols, channels), dtype=numpy.uint8),
        numpy array/cv2 image consisting of pixel values from [0,1] interval
    :param target_width: int target_width of letter boxed image returned
    :param target_height: int target_height of letter boxed image returned
    :return: numpy.ndarray((target_width, target_height, channels), dtype=numpy.uint8
        letterboxed-image with black background, pixels are scaled to be from interval [0,1]
    """

    # Convert gray-scale to RGB
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)

    # Get target ratio for scaling height & width
    im_width, im_height = image.shape[:2]
    height_ratio = target_height / float(im_height)
    width_ratio = target_width / float(im_width)
    ratio = min(height_ratio, width_ratio)

    # Resize image
    image_resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=interpolation)

    # Letter box image
    letter_box = np.full((int(target_width), int(target_height), 3), 0.5)
    row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
    col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
    letter_box[row_start:row_start + image_resized.shape[0], col_start:col_start + image_resized.shape[1], :] = image_resized

    return letter_box
