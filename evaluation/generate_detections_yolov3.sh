#!/usr/bin/env bash

#INPUT_PATH="/Users/martingalajda/School/DIPLOMA-THESIS/firestore-go-utilities/out/images/test1"
# INPUT_PATH=
OUTPUT_PATH="./detections"

if [[ -z "$INPUT_PATH" ]];
then
    echo "INPUT_PATH variable not set!"
else
    echo "Using INPUT_PATH = ${INPUT_PATH}"
    python run_detections.py --input_path=${INPUT_PATH} --output_path=${OUTPUT_PATH} --object_detector=yolov3
fi


