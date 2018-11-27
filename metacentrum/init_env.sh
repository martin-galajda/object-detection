
# cd packages && apt-get download libssl1.0.0 libssl-dev

# ar -x libssl1.0.0_1.0.2l-1~bpo8+1_amd64.deb

# unxz data.tar.xz
# tar -xvf data.tar


# ar -x libssl-dev_1.1.0f-3+deb9u2_amd64.deb; ls

# module add python-3.4.1-intel python34-modules-intel tensorflow-1.7.1-gpu-python3


module add gcc-7.2.0
module add python-3.4.1-gcc


export PYTHONUSERBASE=/software/python34-modules/intel

export PYTHONUSERBASE=/software/python34-modules/gcc
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.4/site-packages:./libs:$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH


module add cuda-8.0
module add cudnn-7.0
module add tensorflow-1.7.1-gpu-python3
module add opencv-3.3.1-py34


pip install numpy --user

cd ~/object-dection && mkdir ./libs && pip3 install h5py -t ./libs

/software/cuda/8.0/lib64/

module add python-3.4.1-intel
export PYTHONUSERBASE=/software/python34-modules/intel
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.4/site-packages:$PYTHONPATH
export PATH=$PYTHONUSERBASE/bin:$PATH

pip install numpy --user
dalsi moduly pipem podle potreby uvedenym zpusobem


kinit; qsub -q gpu -I -l select=1:ncpus=2:ngpus=1:mem=4gb:scratch_local=10gb -l walltime=1:00:00

bashCommand = "qsub -q gpu -I -l select=1:ncpus=2:ngpus=1:mem=4gb:scratch_local=10gb -l walltime=1:00:00"

alias ijob1h="qsub -q gpu -I -l select=1:ncpus=2:ngpus=1:mem=4gb:scratch_local=10gb -l walltime=01:00:00"
alias ijob30m="qsub -q gpu -I -l select=1:ncpus=2:ngpus=1:mem=4gb:scratch_local=10gb -l walltime=0:30:00"
alias ijob10m="qsub -q gpu -I -l select=1:ncpus=2:ngpus=1:mem=4gb:scratch_local=10gb -l walltime=0:10:00"

alias icpujob30m="qsub -I -l select=1:ncpus=2:mem=8gb:scratch_local=20gb -l walltime=0:30:00"
alias icpujob1h="qsub -I -l select=1:ncpus=2:mem=8gb:scratch_local=20gb -l walltime=01:00:00"

alias job="qsub -l select=1:ncpus=1:mem=16gb:scratch_local=20gb -l walltime=23:00:00"