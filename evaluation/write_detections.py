import os
from models.data.base_object_detector import BaseObjectDetector
from models.data.bounding_box import BoundingBox
from typing import List

IMAGE_FILES_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png'
]


def write_image_predictions(
    path_to_output_directory: str,
    filename: str,
    bounding_boxes: List[BoundingBox]
):
    target_out_path = os.path.join(path_to_output_directory, filename + ".txt")

    with open(target_out_path, "w") as out_file:

        for box in bounding_boxes:
            left, top, right, bottom = box.min_x, box.min_y, box.max_x, box.max_y
            class_for_box = box.human_readable_class.lower().replace(" ", "")
            score = box.score

            # <class> <probability> <left> <top> <right> <bottom>
            out_file.write(f'{class_for_box} {str(score)} {int(left)} {int(top)} {int(right)} {int(bottom)}{os.linesep}')


def write_detections(
    path_to_input_directory: str,
    path_to_output_directory: str,
    object_detector: BaseObjectDetector
):
    files_in_dir = os.listdir(path_to_input_directory)

    image_files_in_dir = list(
        filter(lambda filename: any(map(lambda img_ext: img_ext in filename, IMAGE_FILES_EXTENSIONS)), files_in_dir))
    ignored_files = [file_in_dir for file_in_dir in files_in_dir if file_in_dir not in image_files_in_dir]

    print(f'Ignored files({len(ignored_files)}): ')
    print(ignored_files)

    input_dir_name = os.path.split(path_to_input_directory)[1]
    current_out_dir_path = os.path.join(path_to_output_directory, input_dir_name)

    if not os.path.exists(current_out_dir_path):
        os.mkdir(current_out_dir_path)
    current_out_dir_path = os.path.join(current_out_dir_path, object_detector.name)
    if not os.path.exists(current_out_dir_path):
        os.mkdir(current_out_dir_path)

    for image_file_in_dir in image_files_in_dir:
        target_file_path = os.path.join(path_to_input_directory, image_file_in_dir)
        bounding_boxes = object_detector.infer_bounding_boxes_on_target_path(target_file_path)

        write_image_predictions(
            current_out_dir_path,
            image_file_in_dir,
            bounding_boxes
        )

