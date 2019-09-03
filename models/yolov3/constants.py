
class PathConstants:
    YOLOV3_RESOURCES_DIR = './models/yolov3/resources'
    YOLOV3_WEIGHTS_FILE_PATH = f'{YOLOV3_RESOURCES_DIR}/yolov3-openimages.weights'
    YOLOV3_CFG_FILE_PATH = f'{YOLOV3_RESOURCES_DIR}/yolov3-openimages.cfg'

    YOLOV3_MODEL_OPENIMAGES_OUT_PATH = f'{YOLOV3_RESOURCES_DIR}/model.h5'
    YOLOV3_MODEL_OPENIMAGES_GRAPH_OUT_PATH = f'{YOLOV3_RESOURCES_DIR}/graph'


class InferenceStrategies:
    RESIZE_TO_MODEL_SIZE = 'resize_to_model_size'
    LETTER_BOX_TO_MODEL_SIZE = 'letter_box_to_model_size'