#!/usr/bin/env bash

source ~/.bashrc
source ~/.bash_profile

module unload python-3.4.1-gcc
module unload python34-modules-gcc

module load python36-modules-gcc
module load python-3.6.2-gcc

cd ~/object-detection

# export PYTHONPATH=/software/python36-modules/gcc/lib/python3.6/site-packages:./libs-gpu:/software/tensorflow/1.7.1/python34-gpu/lib/python3.4/site-packages

python ./data/openimages/build_images_db.py
