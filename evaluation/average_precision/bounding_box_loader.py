import os
from models.data.bounding_box import BoundingBox


class BoundingBoxLoader:

    @staticmethod
    def loadBoxesFromFiles(
        *,
        path_to_gt_files_dir: str,
        path_to_detected_files_dir: str,
    ):
        gt_boxes = BoundingBoxLoader.loadGtBoxesFromFile(path_to_gt_files_dir)
        detected_boxes = BoundingBoxLoader.loadDetectedBoxesFromFile(path_to_detected_files_dir)

        return gt_boxes, detected_boxes

    @staticmethod
    def loadDetectedBoxesFromFile(path_to_detected_files_dir: str):
        boxes = []

        files_in_dir = os.listdir(path_to_detected_files_dir)

        for file_in_dir in files_in_dir:
            target_file_path = os.path.join(path_to_detected_files_dir, file_in_dir)

            with open(target_file_path) as f:
                for line in f:
                    if line.strip() != "":
                        class_label, score, left, top, right, bottom = line.split()

                        bbox = BoundingBox(
                            min_x=left,
                            min_y=top,
                            max_x=right,
                            max_y=bottom,
                            score=score,
                            class_idx=None,
                            human_readable_class=class_label,
                            filename=file_in_dir
                        )
                        boxes.append(bbox)

        return boxes

    @staticmethod
    def loadGtBoxesFromFile(path_to_gt_files_dir: str):
        gt_boxes = []
        files_in_dir = os.listdir(path_to_gt_files_dir)

        for file_in_dir in files_in_dir:
            target_file_path = os.path.join(path_to_gt_files_dir, file_in_dir)

            with open(target_file_path) as f:
                for line in f:
                    if line.strip() != "":
                        class_label, left, top, right, bottom = line.split()

                        bbox = BoundingBox(
                            min_x=left,
                            min_y=top,
                            max_x=right,
                            max_y=bottom,
                            score=1.,
                            class_idx=None,
                            human_readable_class=class_label,
                            filename=file_in_dir
                        )
                        gt_boxes.append(bbox)

        return gt_boxes
