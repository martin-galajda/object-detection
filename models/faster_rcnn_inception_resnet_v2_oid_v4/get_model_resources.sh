#!/usr/bin/env bash

echo "Getting model resources. This can take a while..."
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz

echo "Extracting model resources to temp directory"
mkdir ./tmp_resources
tar -xzvf faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz -C ./tmp_resources

echo "Copying model resources from temp directory"
cp ./tmp_resources/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb ./resources/frozen_inference_graph.pb

echo "Deleting temp resources"
rm -r ./tmp_resources
rm faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz