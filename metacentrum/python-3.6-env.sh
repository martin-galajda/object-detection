#!/usr/bin/env bash

cd /storage/brno3-cerit/home/marneyko/object-detection

module unload cuda-8.0
module unload tensorflow-1.7.1-gpu-python3
module unload python-3.4.1-gcc
module unload python34-modules-gcc

module load cuda-9.0
module load python36-modules-gcc
module load python-3.6.2-gcc

export LD_LIBRARY_PATH=./tmp-cudnn/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export PYTHONPATH=`pwd`:$PYTHONPATH

source ./venv/bin/activate
