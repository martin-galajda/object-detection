#!/usr/bin/env bash

cd /storage/brno3-cerit/home/marneyko/object-detection

module load python36-modules-gcc
module load python-3.6.2-gcc

source ./venv/bin/activate
export PYTHONPATH=`pwd`:$PYTHONPATH

cd /storage/brno3-cerit/home/marneyko/object-detection &&
    module load python36-modules-gcc python-3.6.2-gcc &&
    source ./venv/bin/activate &&
    export PYTHONPATH=`pwd`:$PYTHONPATH &&
    python data/openimages/build_labels_db.py --path_to_image_labels_file=./data/openimages/train-annotations-human-imagelabels.csv --table=image_labels