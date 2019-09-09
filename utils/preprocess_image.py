import cv2
import numpy as np
import PIL


def preprocess_image_bytes(image_bytes):
    return image_bytes / 255.0


def resize_and_letter_box(image, *, target_width: int, target_height: int):
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
    im_width, im_height = image.shape[:2]
    height_ratio = target_height / float(im_height)
    width_ratio = target_width / float(im_width)
    ratio = min(height_ratio, width_ratio)

    image_resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

    letter_box = np.full((int(target_width), int(target_height), 3), 0.5)
    row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
    col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
    letter_box[row_start:row_start + image_resized.shape[0], col_start:col_start + image_resized.shape[1], :] = image_resized

    return letter_box


def resize_and_letter_box_pil_image(image: PIL.Image, *, target_width: int, target_height: int):
    """
    Letter box (black bars) a color image (think pan & scan movie shown
    on widescreen) if not same aspect ratio as specified rows and cols.
    :param image: PIL.Image
    :param target_width: int target_width of letter boxed image returned
    :param target_height: int target_height of letter boxed image returned
    :return: numpy.ndarray((rows, cols, channels), dtype=numpy.uint8)
    """
    img_width, img_height = np.asarray(image).shape[:2]
    width_ratio = target_width / float(img_width)
    height_ratio = target_height / float(img_height)
    ratio = min(width_ratio, height_ratio)

    image_resized = image.resize((int(img_width * ratio), int(img_height * ratio)))
    image_resized = np.asarray(image_resized)

    letter_box = np.full((int(target_width), int(target_height), 3), 0.5)
    row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
    col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
    letter_box[row_start:row_start + image_resized.shape[0], col_start:col_start + image_resized.shape[1], :] = image_resized

    return letter_box


def resize_and_letter_box_pil_image_2(image: PIL.Image, *, target_width: int, target_height: int):
    """
    Letter box (black bars) a color image (think pan & scan movie shown
    on widescreen) if not same aspect ratio as specified rows and cols.
    :param image: PIL.Image
    :param target_width: int target_width of letter boxed image returned
    :param target_height: int target_height of letter boxed image returned
    :return: numpy.ndarray((rows, cols, channels), dtype=numpy.uint8)
    """
    iw, ih = image.size
    w, h = target_width, target_height
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), PIL.Image.BICUBIC)
    new_image = PIL.Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def resize_img(bytes_img: np.array, target_img_size, letterbox_image=False):
    if letterbox_image:
        img_numpy = np.fromstring(bytes_img, np.uint8)
        img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        img_arr = img_cv2.astype(np.uint8)
        img = resize_and_letter_box(img_arr, target_height = target_img_size[1], target_width = target_img_size[0])
        img = np.array(img, dtype=np.uint8)
    else:
        img_numpy = np.fromstring(bytes_img, np.uint8)

        img_cv2 = cv2.imdecode(img_numpy, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        img_resized = cv2.resize(img_cv2, target_img_size)
        img = img_resized.astype(np.uint8)

    return img
