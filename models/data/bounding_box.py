from typing import List, Any, Dict


class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    score: float
    class_idx: int
    human_readable_class: str
    filename: str

    def __init__(
        self,
        *,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        score: float,
        class_idx: int,
        human_readable_class: str,
        filename: str = None
    ):
        self.min_x = int(min_x)
        self.min_y = int(min_y)
        self.max_x = int(max_x)
        self.max_y = int(max_y)
        self.class_idx = class_idx
        self.score = float(score)
        self.human_readable_class = human_readable_class
        self.filename = filename


def make_bounding_boxes(
    *,
    inferred_boxes: List[Any],
    inferred_classes: List[Any],
    inferred_scores: List[float],
    class_index_to_human_readable_dict: Dict[int, str]
) -> List[BoundingBox]:
    """
    Makes list of Bounding Box classes from raw outputs produced by the CNN for object detection.

    :param inferred_boxes: boxes produced by the CNN (absolute with respect to original image dimensions)
    :param inferred_classes: class indices inferred by the CNN
    :param inferred_scores: probability scores inferred by the CNN
    :param class_index_to_human_readable_dict: dictionary containing mapping from label idx to its human readable form
    :return:list of Bounding Box objects
    """

    bounding_boxes = []

    for i in range(len(inferred_boxes)):
        min_x, min_y, max_x, max_y = inferred_boxes[i]
        class_idx = inferred_classes[i]
        score = inferred_scores[i]
        human_readable_class = class_index_to_human_readable_dict[class_idx]
        bounding_boxes.append(BoundingBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            score=score,
            human_readable_class=human_readable_class,
            class_idx=class_idx
        ))

    return bounding_boxes
