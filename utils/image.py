import numpy as np
from PIL import Image


def load_pil_image_from_file(file_path: str) -> (Image, np.array):
    """
    Loads image specified by 'file_path' as a RGB PIL image.

    Args:
      - file_path: path to the image to be loaded

    Returns:
      Tuple consting of PIL Image instance nad numpy array containing pixels in row-major order.
    """
    img = Image.open(file_path)
    img.load()
    rgb_image = img.convert('RGB')

    return rgb_image, np.asarray(rgb_image, dtype="int32")
