import numpy as np

def non_max_suppression(boxes: list, scores: list, box_classes: list, min_iou: float):
    def remove_elements_by_indices(arr: list, indices: list):
        return [elem for i, elem in enumerate(arr) if i not in indices]

    def compute_ious(box, list_of_boxes):
        min_x, min_y, max_x, max_y = box[:4]
        area_box = [max_x - min_x, max_y - min_y]
        width_boxes = list_of_boxes[:, 2] - list_of_boxes[:, 0]
        height_boxes = list_of_boxes[:, 3] - list_of_boxes[:, 1]

        possible_intersection_beginnings_x = np.maximum(min_x, list_of_boxes[:, 0])
        possible_intersection_beginnings_y = np.maximum(min_y, list_of_boxes[:, 1])

        possible_intersection_ends_x = np.minimum(max_x, list_of_boxes[:, 2])
        possible_intersection_ends_y = np.minimum(max_y, list_of_boxes[:, 3])

        intersection_areas_widths = np.maximum(possible_intersection_ends_x - possible_intersection_beginnings_x, 0)
        intersection_areas_heights = np.maximum(possible_intersection_ends_y - possible_intersection_beginnings_y, 0)
        intersection_areas = intersection_areas_widths * intersection_areas_heights

        union_areas = np.array([area_box[0] * area_box[1]])
        union_areas = union_areas + width_boxes * height_boxes

        union_areas = union_areas - intersection_areas

        ious = intersection_areas / union_areas

        return ious

    assert len(boxes) == len(scores), "len(boxes) == len(scores) for non max suppression"

    # sorted ascendingly by score, we will sequentially pop last elements (highest scores)
    sorted_scores_indices = np.array(scores).argsort().tolist()
    boxes = np.array(boxes)

    indices_for_filtered_boxes = []
    while len(sorted_scores_indices) > 0:
        current_highest_score_idx = sorted_scores_indices.pop()
        box_indices_without_current_box = sorted_scores_indices

        ious = compute_ious(boxes[current_highest_score_idx], boxes[box_indices_without_current_box])

        box_indices_with_significant_iou = np.where(ious >= min_iou)[0]
        current_box_class = box_classes[current_highest_score_idx]

        box_indices_to_remove = filter(lambda box_index: current_box_class == box_classes[box_index], box_indices_with_significant_iou)
        sorted_scores_indices = remove_elements_by_indices(sorted_scores_indices, box_indices_to_remove)

        indices_for_filtered_boxes += [current_highest_score_idx]

    return indices_for_filtered_boxes


def classic_non_max_suppression(boxes: list, scores: list, min_iou: float):
    def remove_elements_by_indices(arr: list, indices: list):
        return [elem for i, elem in enumerate(arr) if i not in indices]

    def compute_ious(box, list_of_boxes):
        min_x, min_y, max_x, max_y = box[:4]
        area_box = [max_x - min_x, max_y - min_y]
        width_boxes = list_of_boxes[:, 2] - list_of_boxes[:, 0]
        height_boxes = list_of_boxes[:, 3] - list_of_boxes[:, 1]

        possible_intersection_beginnings_x = np.maximum(min_x, list_of_boxes[:, 0])
        possible_intersection_beginnings_y = np.maximum(min_y, list_of_boxes[:, 1])

        possible_intersection_ends_x = np.minimum(max_x, list_of_boxes[:, 2])
        possible_intersection_ends_y = np.minimum(max_y, list_of_boxes[:, 3])

        intersection_areas_widths = np.maximum(possible_intersection_ends_x - possible_intersection_beginnings_x, 0)
        intersection_areas_heights = np.maximum(possible_intersection_ends_y - possible_intersection_beginnings_y, 0)
        intersection_areas = intersection_areas_widths * intersection_areas_heights

        union_areas = np.array([area_box[0] * area_box[1]])
        union_areas = union_areas + width_boxes * height_boxes

        union_areas = union_areas - intersection_areas

        ious = intersection_areas / union_areas

        return ious

    assert len(boxes) == len(scores), "len(boxes) == len(scores) for non max suppression"

    # sorted ascendingly by score, we will sequentially pop last elements (highest scores)
    sorted_scores_indices = np.array(scores).argsort().tolist()
    boxes = np.array(boxes)

    indices_for_filtered_boxes = []
    while len(sorted_scores_indices) > 0:
        current_highest_score_idx = sorted_scores_indices.pop()
        box_indices_without_current_box = sorted_scores_indices

        ious = compute_ious(boxes[current_highest_score_idx], boxes[box_indices_without_current_box])

        box_indices_to_remove = np.where(ious >= min_iou)[0]
        sorted_scores_indices = remove_elements_by_indices(sorted_scores_indices, box_indices_to_remove)

        indices_for_filtered_boxes += [current_highest_score_idx]

    return indices_for_filtered_boxes

# Malisiewicz et al.
def non_max_suppression_fast(boxes: np.array, overlap_thresh: float):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
