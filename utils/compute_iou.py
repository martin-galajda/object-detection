import numpy as np

def compute_iou(box_one, box_two):
    """
    
    Parameters:
        - box_one - numpy array or list, Possible shape: [[x1, y1], [x2, y2]] or [x1, y1, x2,  y2] where:
            - x1, y1 are top-left coordinates of box
            - x2, y2 are bottom-right coordinates of box
        - box_two - [[x1, y1], [x2, y2]] or [x1, y1, x2, y2]
            - x1, y1 are top-left coordinates of box
            - x2, y2 are bottom-right coordinates of box
    Returns:
        - iou - float, Intersection over union between two boxes
    """
    
    box_one = np.array(box_one).reshape((2, 2))
    box_two = np.array(box_two).reshape((2, 2))
    
    box_maxes = np.maximum(box_one, box_two)
    box_mins = np.minimum(box_one, box_two)
    
    # compute the point defining start of possible intersection area
    start_intersection = box_maxes[0]
    
    # compute the point defining end of possible intersection area
    end_intersection = box_mins[1]
    
    # compute the difference (length from start to the end - 2 dimensions - width & height)
    end_minus_start = end_intersection - start_intersection
    
    # clip minimum value to be 0 (if there is no intersection the values are < 0)
    intersection = np.clip(end_minus_start, 0, None)
    
    # compute intersection area
    intersection_area = intersection[0] * intersection[1]
    
    # compute box areas
    box_one_area = (box_one[1][0]  - box_one[0][0]) * (box_one[1][1]  - box_one[0][1])
    box_two_area = (box_two[1][0]  - box_two[0][0]) * (box_two[1][1]  - box_two[0][1])

    # compute union of both boxes
    box_union_area = box_one_area + box_two_area - intersection_area

    # compute intersection over area
    iou = intersection_area / box_union_area
    
    return iou
