# What is it?

This repository contains all the code for Diploma thesis in which we use object detection models on OpenImages dataset to solve problem of extracting  meaningful information from the image content present on the websites.

# Initial setup
In order to be able to run object detectors we need to download and generate resources needed by the inference pipelines.

## Setuping Faster R-CNN

Follow the instructions [here](https://github.com/martinGalajdaSchool/object-detection/tree/master/models/faster_rcnn_inception_resnet_v2_oid_v4).

## Setuping YOLOv3

Follow the instructions [here](https://github.com/martinGalajdaSchool/object-detection/tree/master/models/yolov3/README.md).


# Directory structure

Our repository is structured into multiple repositories:

- /models - code for object detection inference pipelines
- /models/data  - data structures used by object detection algorithms
- /models/utils - utility functions used by object detection algorithms
- /evaluation - code for computing evaluation metrics for object detection models
- /notebooks - Jupyter notebooks which
  - demonstrate how to use the object detectors
  - contain scripts used for computing metrics
- /utils - common utility functions
- /common - common stuff i.e. custom argparse types

# Object detectors

After initial setup of models, you can use object detectors in the way described below.


## YOLOv3

```
from models.yolov3.object_detector import ObjectDetector as YOLOv3ObjectDetector

yolov3_detector = YOLOv3ObjectDetector(
    # Be verbose about times about inference
    verbose=True,

    # Enable logging of the device placement
    log_device_placement=True,

    # Control probability threshold for detecting objects
    detection_threshold=0.3,

    # Control threshold for non-max suppression
    nms_threshold=0.6
)

target_path = ... # specify target file path to the image
bounding_boxes = yolov3_detector.infer_bounding_boxes_on_target_path(target_path)

# Alternatively you can infer bounding boxes by loaded numpy image (watch out for row-major and col-major ordering issues!)
np_image = ... # load RGB image into numpy image in row-major ordering
bounding_boxes = yolov3_detector.infer_bounding_boxes_on_loaded_image(np_image)

```

## Faster R-CNN
```
from models.faster_rcnn_inception_resnet_v2_oid_v4.object_detector import ObjectDetector as FasterRCNNObjectDetector

faster_rcnn_detector = FasterRCNNObjectDetector(
    # Enable use of the GPU
    use_gpu=True,

    # Enable logging of the device placement
    log_device_placement=True
)

target_path = ... # specify target file path to the image
bounding_boxes = faster_rcnn_detector.infer_bounding_boxes_on_target_path(target_path)

# Alternatively, you can infer bounding boxes by loaded numpy image (watch out for row-major and col-major ordering issues!)
np_image = ... # load RGB image into numpy image in row-major ordering
bounding_boxes = faster_rcnn_detector.infer_bounding_boxes_on_loaded_image(np_image)

```


# Technologies used
- TensorFlow
- Keras
- Scrapy
- Jupyter
- Pillow
