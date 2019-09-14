import argparse
from models.yolov3_gpu_head_v2.conversion.darknet_model_to_keras import convert_model
from models.yolov3_gpu_head_v2.constants import PathConstants


class CLIArgs:
    yolov3_darknet_config_path = 'yolov3_darknet_config_path'
    yolov3_darknet_weights_path = 'yolov3_darknet_weights_path'
    yolov3_converted_keras_model_path = 'yolov3_converted_keras_model_path'
    yolov3_out_graph_path = 'yolov3_out_graph_path'


def main(arguments: argparse.Namespace):
    convert_model(
        getattr(arguments, CLIArgs.yolov3_darknet_config_path, None),
        getattr(arguments, CLIArgs.yolov3_darknet_weights_path, None),
        getattr(arguments, CLIArgs.yolov3_converted_keras_model_path, None),
        fully_convolutional=True,
        plot_model=True,
        path_to_graph_output=getattr(arguments, CLIArgs.yolov3_out_graph_path, None)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script converting YOLOv3 model trained on OpenImages in Darknet framework to keras model."
    )

    parser.add_argument(f'--{CLIArgs.yolov3_darknet_config_path}',
                        type=str,
                        default=PathConstants.YOLOV3_CFG_FILE_PATH,
                        help='Path to config file for YOLOv3 model trained in Darknet framework on OpenImages dataset.')

    parser.add_argument(f'--{CLIArgs.yolov3_darknet_weights_path}',
                        type=str,
                        default=PathConstants.YOLOV3_WEIGHTS_FILE_PATH,
                        help='Path to file containing trained weight for YOLOv3 model trained in Darknet framework on OpenImages dataset.')

    parser.add_argument(f'--{CLIArgs.yolov3_converted_keras_model_path}',
                        type=str,
                        default=PathConstants.YOLOV3_MODEL_OPENIMAGES_OUT_PATH,
                        help='Destination path for writing converted keras YOLOv3 model.')

    parser.add_argument(f'--{CLIArgs.yolov3_out_graph_path}',
                        type=str,
                        default=PathConstants.YOLOV3_MODEL_OPENIMAGES_GRAPH_OUT_PATH,
                        help='Destination path for writing graph of YOLOv3 model.')

    args = parser.parse_args()
    main(args)
