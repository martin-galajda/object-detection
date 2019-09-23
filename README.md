# What is it?

This repository contains all the code for Diploma thesis in which we use object detection models on [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) to tackle the problem of extracting useful information from the image content present on the web.

# Initial setup
In order to be able to run object detectors we need to download and generate resources needed by the inference pipelines. Moreover, to maintain reproducible environment, we recommend using virtualenv.

## Prerequisites

1. [Python 3.6](https://www.python.org/downloads/)


## Setuping virtual environment

1. `pip install venv`  <-- if you don't have venv installed yet
2. `python -m virtualenv venv` <-- creates a virtual environment in the venv directory
3. `source venv/bin/activate` <-- activates the virtual environment
4. `pip install -r requirements.txt` <-- installs packages into the virtual environment
5. `./set_python_path.sh` <-- IMPORTANT: We write all Python code assuming `PYTHONPATH` is pointing to the root directory

## Setuping Faster R-CNN

Follow the instructions [here](https://github.com/martinGalajdaSchool/object-detection/tree/master/models/faster_rcnn_inception_resnet_v2_oid_v4).

## Setuping YOLOv3

Follow the instructions [here](https://github.com/martinGalajdaSchool/object-detection/tree/master/models/yolov3/README.md).


# Directory structure

Our repository is structured into multiple folders:

- `/models` - code for object detection inference pipelines
- `/models/data`  - data structures used by object detection algorithms
- `/models/utils` - utility functions used by object detection algorithms
- `/models/preprocessing` - preprocessing used in the models
- `/models/yolov3` - YOLOv3 inference pipeline:
  - `/cpu_head/` - with inference head in Numpy (on CPU)
  - `/gpu_head_v1` - with inference head in TF, but non-max suppression in Numpy
  - `/gpu_head_v2` - with inference head and non-max suppression in TF
  - `/conversion` - conversion from the [Darknet framework](https://github.com/pjreddie/darknet) to Keras
  - `/resources` - resources needed by the model (e.g. weights, class labels)

- `/models/faster_rcnn_inception_resnet_v2_oid_v4`- Faster R-CNN inference pipeline
- `/evaluation` - code for computing evaluation metrics for object detection models
- `/evaluation/average_precision` - our custom implementation for computing (m)AP 
- `/notebooks` - Jupyter notebooks which
  - demonstrate how to use the object detectors
  - contain scripts used for computing metrics
- `/utils` - common utility functions
- `/common` - common stuff i.e. custom argparse types

# Object detectors

After initial setup of models, you can use object detectors in the way described below.


## YOLOv3

```
from models.yolov3.object_detector import ObjectDetector as YOLOv3ObjectDetector

yolov3_detector = YOLOv3ObjectDetector(
    # Be verbose about times for inference
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
