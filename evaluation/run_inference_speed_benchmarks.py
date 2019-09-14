from models.faster_rcnn_inception_resnet_v2_oid_v4.object_detector import ObjectDetector as FasterRCNNObjectDetector
from models.yolov3_gpu_head.object_detector import ObjectDetector as YOLOv3ObjectDetector
from evaluation.speed_tester import SpeedTester

faster_rcnn_detector = FasterRCNNObjectDetector(None, None, None, True)
yolov3_object_detector = YOLOv3ObjectDetector()

PATH_TO_TEST_IMAGE = './resources/horse.jpeg'

speed_tester = SpeedTester(PATH_TO_TEST_IMAGE)

print(f'*** Benchmarking YOLOv3: START ***')
time_yolov3 = speed_tester.benchmark_detector(yolov3_object_detector, 10)
print(f'*** Benchmarking YOLOv3: END ***')

print(f'*** Benchmarking Faster R-CNN: START ***')
time_faster_rcnn = speed_tester.benchmark_detector(faster_rcnn_detector, 10)
print(f'*** Benchmarking Faster R-CNN: END ***')

print(f'YOLOv3 avg inference speed: {time_yolov3} seconds.')
print(f'Faster R-CNN avg inference speed: {time_faster_rcnn} seconds.')
