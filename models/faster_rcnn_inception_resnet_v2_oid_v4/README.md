# Prerequisites:

1. Protobuf compiler (protoc) - look [here](https://github.com/protocolbuffers/protobuf) for more info on how to install.
2. Python3 virtual environment enabled  (from the root run `source ./venv/bin/activate`).

# How to setup model resources

From  `/models/faster_rcnn_inception_resnet_v2_oid_v4` directory:
1. Run `. get_model_resources.sh`
2. Run `cd ./protos && . compile.sh`