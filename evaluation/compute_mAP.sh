#!/usr/bin/env

DIR_BEFORE_SCRIPT=$(pwd)
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OBJ_DETECTION_METRICS_DIR="${CURR_DIR}/Object-Detection-Metrics/"
MODEL_NAME=$1

# EXPORT_FILENAME="export-2019-09-05T09_32_41.906Z"
EXPORT_FILENAME=

if [ -z $EXPORT_FILENAME ];
then
    echo "EXPORT_FILENAME variable not set!" && exit 0
fi


PATH_TO_DETECTIONS_DIR="${CURR_DIR}/detections/${EXPORT_FILENAME}/${MODEL_NAME}"
PATH_TO_GROUND_TRUTH_DIR="${CURR_DIR}/ground_truth/${EXPORT_FILENAME}"
OUT_DIR="${CURR_DIR}/results/${MODEL_NAME}"

mkdir -p $OUT_DIR

if [ -d "$OBJ_DETECTION_METRICS_DIR" ]; then
    echo "Directory $OBJ_DETECTION_METRICS_DIR already exists"
else
    echo "Directory $OBJ_DETECTION_METRICS_DIR does not exist. Cloning..."
    git clone https://github.com/rafaelpadilla/Object-Detection-Metrics.git
fi

echo "Generating evaluation results..."
echo "Ground truth folder path = ${PATH_TO_GROUND_TRUTH_DIR}"
echo "Detections folder path = ${PATH_TO_DETECTIONS_DIR}"


cd ${OBJ_DETECTION_METRICS_DIR} \
    && python3 pascalvoc.py \
        --gtfolder=${PATH_TO_GROUND_TRUTH_DIR} \
        --detfolder=${PATH_TO_DETECTIONS_DIR} \
        --savepath=${OUT_DIR}

cd $DIR_BEFORE_SCRIPT
echo "Evaluation results generated..."

