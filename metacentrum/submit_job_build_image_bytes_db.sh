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
    python data/openimages/add_image_data.py --num_of_images_to_process=500001 --start_idx=2500000 --batch_process_count=100

# TRAIN - highest is so far 2500000 --> 3000001 (So 3 mil is imported for train)
