import numpy as np
import PIL
from PIL import ImageDraw
from typing import List
from models.data.bounding_box import BoundingBox


def draw_detected_boxes_on_pil_image(detected_boxes: np.array, detected_classes: list, detected_scores: np.array, img: PIL.Image):
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = box
        curr_detected_class_names = detected_classes[idx]
        curr_detected_scores = detected_scores[idx]

        classes_with_scores = list(map(lambda x: f'{x[0]}({int(x[1] * 100)}%)', zip(curr_detected_class_names,  curr_detected_scores)))
        draw_rectangle(draw, [x1, y1, x2, y2])
        print(curr_detected_class_names)
        draw.text(((max(x1, 0)), (y1) - 10), ','.join(classes_with_scores), fill="red")


def draw_detected_boxes_on_pil_image_v2(detected_boxes: np.array, detected_classes: list, detected_scores: np.array, img: PIL.Image):
    draw = ImageDraw.Draw(img)
    for idx, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = box
        curr_detected_class_names = detected_classes[idx]
        curr_detected_scores = detected_scores[idx]

        classes_with_scores = f'{curr_detected_class_names} {int(curr_detected_scores * 100)}%'
        draw_rectangle(draw, [x1, y1, x2, y2])
        print(curr_detected_class_names)
        draw.text(((max(x1, 0)), (y1) - 10), classes_with_scores, fill="red")


def draw_bounding_boxes_on_pil_image(bounding_boxes: List[BoundingBox], img: PIL.Image):
    draw = ImageDraw.Draw(img)
    for box in bounding_boxes:
        classes_with_scores = f'{box.human_readable_class} {int(box.score * 100)}%'
        draw_rectangle(draw, [box.min_x, box.min_y, box.max_x, box.max_y])
        draw.text(((max(box.min_x, 0)), (box.min_y) - 10), classes_with_scores, fill="red")


def draw_rectangle(draw_context: ImageDraw, box_coordinates: np.array, color="red", width=3):
    x1, y1, x2, y2 = box_coordinates
    offset = 1
    for i in range(0, width):
        draw_context.rectangle(((x1, y1), (x2, y2)), outline=color)
        x1 = x1 - offset
        y1 = y1 + offset
        x2 = x2 + offset
        y2 = y2 - offset
