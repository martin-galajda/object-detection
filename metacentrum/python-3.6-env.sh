#!/usr/bin/env bash

export PYTHONUSERBASE=/software/python-3.6.2/gcc
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.6/site-packages:./libs:$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH

# module unload ss
module unload python-3.4.1-gcc
module unload python34-modules-gcc
module load python-3.6.2-gcc

cd ~/object-detection
pip3 install opencv-python aiohttp -t ./libs

python dataset_parsers.py
