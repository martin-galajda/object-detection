import tensorflow as tf
from models.faster_rcnn_inception_resnet_v2_oid_v4.protos import string_int_label_map_pb2
from google.protobuf import text_format
from models.faster_rcnn_inception_resnet_v2_oid_v4.constants import FasterRCNNPathConstants


def get_labels_map(path_to_labels_pb_file: str = FasterRCNNPathConstants.PATH_TO_LABELS_PB_TXT):
    with tf.gfile.GFile(path_to_labels_pb_file, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    return label_map


def build_class_index(path_to_labels_pb_file: str = FasterRCNNPathConstants.PATH_TO_LABELS_PB_TXT) -> dict:
    """Builds index for mapping integer number to human readable classes.

    :param path_to_labels_pb_file: Optional path to where .pbtxt file for labels is contained
    :return: Dictionary containing mapping from integer indices to human readable classes.
    """
    class_index = {}

    labels_map = get_labels_map(path_to_labels_pb_file)
    for label_map_item in labels_map.item:
        class_index[label_map_item.id] = label_map_item.display_name

    return class_index


