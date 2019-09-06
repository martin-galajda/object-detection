import os

IMAGE_FILES_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png'
]


def write_image_predictions(
    path_to_output_directory: str,
    filename: str,
    predictions: tuple
):
    target_out_path = os.path.join(path_to_output_directory, filename + ".txt")
    detected_boxes, detected_classes, detected_scores = predictions

    with open(target_out_path, "w") as out_file:

        for i, detected_box in enumerate(detected_boxes):
            left, top, right, bottom = detected_box
            classes_for_box = detected_classes[i]

            for j, class_for_box in enumerate(classes_for_box):
                score = detected_scores[i][j]

                # <class> <probability> <left> <top> <right> <bottom>
                out_file.write(f'{str(class_for_box).lower().replace(" ", "")} {str(score)} {int(left)} {int(top)} {int(right)} {int(bottom)}{os.linesep}')


def write_detections(
    path_to_input_directory: str,
    path_to_output_directory: str,
    object_detector
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
        predictions = object_detector.infer_object_detections(target_file_path)

        write_image_predictions(current_out_dir_path,
                                image_file_in_dir,
                                predictions)

