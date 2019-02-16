import numpy as np

from models.yolov2.constants import Constants


def load_anchors(path_prefix = ''):
    with open(path_prefix + Constants.PATH_TO_YOLO_V2_COCO_ANCHORS, 'r') as f:
        anchors_line = f.readline()
        anchors = list(map(lambda x: float(x.strip()), anchors_line.split(',')))

        return np.array(anchors).reshape((5, 2))
