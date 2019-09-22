from models.faster_rcnn_inception_resnet_v2_oid_v4.object_detector import ObjectDetector as FasterRCNNObjectDetector
from models.yolov3.object_detector import ObjectDetector as YOLOv3ObjectDetector
from evaluation.speed_tester import SpeedTester

faster_rcnn_detector = FasterRCNNObjectDetector(use_gpu=True, log_device_placement=False)
yolov3_object_detector = YOLOv3ObjectDetector(log_device_placement=False)

PATH_TO_TEST_IMAGE = './resources/horse.jpeg'

speed_tester = SpeedTester(PATH_TO_TEST_IMAGE)

print(f'*** Benchmarking YOLOv3: START ***')
time_yolov3 = speed_tester.benchmark_detector(yolov3_object_detector, 20)
print(f'*** Benchmarking YOLOv3: END ***')

print(f'*** Benchmarking Faster R-CNN: START ***')
time_faster_rcnn = speed_tester.benchmark_detector(faster_rcnn_detector, 20)
print(f'*** Benchmarking Faster R-CNN: END ***')

print(f'YOLOv3 avg inference speed: {time_yolov3} seconds.')
print(f'Faster R-CNN avg inference speed: {time_faster_rcnn} seconds.')
