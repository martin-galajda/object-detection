from models.yolov2.utils.convert_darknet_to_keras_model import darknet_yolov2_to_keras
from models.yolov2.constants import Constants
from common.argparse_types import str2bool
import argparse

def main(args):
    output_path = None
    if args.fully_conv:
        output_path = Constants.PATH_TO_PRETRAINED_YOLO_V2_MODEL_FULLY_CONV
    else:
        output_path = Constants.PATH_TO_PRETRAINED_YOLO_V2_MODEL
    fully_conv = args.fully_conv

    darknet_yolov2_to_keras(
        args.path_to_yolo_v2_cfg,
        args.path_to_yolo_v2_coco_weights,
        output_path,
        fully_convolutional=fully_conv,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script responsible for converting pretrained coco YOLOv2 to keras model.'
    )

    parser.add_argument('--fully_conv',
                        type=str2bool,
                        default=False,
                        required=False)

    parser.add_argument('--path_to_yolo_v2_cfg',
                        type=str,
                        default=Constants.PATH_TO_YOLO_V2_CFG,
                        required=False)

    parser.add_argument('--path_to_yolo_v2_coco_weights',
                        type=str,
                        default=Constants.PATH_TO_YOLO_V2_COCO_WEIGHTS,
                        required=False)

    main(parser.parse_args())
