import numpy as np
from PIL import Image
import cv2


def load_pil_image_from_file(file_path: str) -> (Image, np.array):
    img = Image.open(file_path)
    img.load()
    return img, np.asarray(img, dtype="int32")


def load_image_from_file(file_path: str):
    img = cv2.imread(file_path)
    return img


def resize_pil_image(img: Image, *, width: int, height:  int) -> (Image, np.array):
    resized_img = img.resize((width, height))
    return resized_img, np.asarray(resized_img, dtype="int32")
