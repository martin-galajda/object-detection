import os

SCRIPT_PATH_TOKENS = os.path.realpath(__file__).split(os.sep)
ROOT_DIR = os.sep.join(SCRIPT_PATH_TOKENS[:len(SCRIPT_PATH_TOKENS)-1])


class PathConstants:
    PATH_TO_DETECTIONS_DIR = f'{ROOT_DIR}/detections/'