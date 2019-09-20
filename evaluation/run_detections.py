import argparse
from models.yolov3_gpu_head_v2.object_detector import ObjectDetector as YOLOv3ObjectDetector
from models.faster_rcnn_inception_resnet_v2_oid_v4.object_detector import ObjectDetector as FasterRCNNObjectDetector
from evaluation.write_detections import write_detections


class ObjectDetectorOptions:
    YOLOV3 = 'yolov3'
    FASTER_RCNN = 'faster_rcnn'


def generate_detections_files(
    input_path: str,
    output_path: str,
    object_detector: str,
    detector_constructor_args: dict = {}
):
    model = YOLOv3ObjectDetector(**detector_constructor_args) \
        if object_detector == ObjectDetectorOptions.YOLOV3 \
        else FasterRCNNObjectDetector(**detector_constructor_args)

    write_detections(input_path, output_path, model)


def main():
    parser = argparse.ArgumentParser(description='Generate detection files required for computing mAP.')

    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Input path to directory containing images for evaluation.')

    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Output path to directory where detections will be written.')

    parser.add_argument('--object_detector',
                        type=str,
                        choices=[ObjectDetectorOptions.YOLOV3, ObjectDetectorOptions.FASTER_RCNN],
                        required=True,
                        help=f'Determine which object detector to use ({ObjectDetectorOptions.YOLOV3}, {ObjectDetectorOptions.FASTER_RCNN}).')

    args = parser.parse_args()
    generate_detections_files(
        input_path=args.input_path,
        output_path=args.output_path,
        object_detector=args.object_detector
    )


if __name__ == "__main__":
    main()
