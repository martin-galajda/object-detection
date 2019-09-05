import argparse
from models.yolov3.object_detector import ObjectDetector as YOLOv3ObjectDetector
from models.faster_rcnn_inception_resnet_v2_oid_v4.object_detector import ObjectDetector as FasterRCNNObjectDetector
from evaluation.write_detections import write_detections


class ObjectDetectorOptions:
    YOLOV3 = 'yolov3'
    FASTER_RCNN = 'faster_rcnn'


def generate_detections_files(args):
    model = YOLOv3ObjectDetector() \
        if args.object_detector == ObjectDetectorOptions.YOLOV3 \
        else FasterRCNNObjectDetector()

    write_detections(args.input_path, args.output_path, model)


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
                        help='Determine which object detector to use (YOLOV3/FasterRCNN).')

    args = parser.parse_args()
    generate_detections_files(args)


if __name__ == "__main__":
    main()
