import os

SCRIPT_PATH_TOKENS = os.path.realpath(__file__).split(os.sep)
ROOT_DIR = os.sep.join(SCRIPT_PATH_TOKENS[:len(SCRIPT_PATH_TOKENS)-1])


class FasterRCNNPathConstants:
    PATH_TO_FROZEN_TF_GRAPH = f'{ROOT_DIR}/resources/frozen_inference_graph.pb'
    PATH_TO_LABELS_PB_TXT = f'{ROOT_DIR}/resources/oid_v4_label_map.pbtxt'
