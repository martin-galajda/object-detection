from models.yolov3.conversion.utils import load_classes
import PIL
import numpy as np
from PIL import ImageDraw


def draw_yolov3_detected_boxes(detected_boxes: np.array, detected_classes: np.array, detected_scores: np.array, img: PIL.Image, path_to_classes_file: str = None):
    draw = ImageDraw.Draw(img)
    class_idx_to_class_name = load_classes(path_to_classes_file)
    for idx, box in enumerate(detected_boxes):
        x1, y1, x2, y2, _ = box
        detected_classes_indices = detected_classes[idx]
        draw_rectangle(draw, [x1, y1, x2, y2])
        detected_class_names = list(map(lambda idx: class_idx_to_class_name[idx], detected_classes_indices))

        print(detected_class_names)
        draw.text((((x1 + x2) // 2) - 5, ((y1 + y2) // 2) - 5), ','.join(detected_class_names), fill="red")


def draw_rectangle(draw_context: ImageDraw, box_coordinates: np.array, color="red", width=3):
    x1, y1, x2, y2 = box_coordinates
    offset = 1
    for i in range(0, width):
        draw_context.rectangle(((x1, y1), (x2, y2)), outline=color)
        x1 = x1 - offset
        y1 = y1 + offset
        x2 = x2 + offset
        y2 = y2 - offset
