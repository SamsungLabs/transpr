#!/bin/bash

conda create -y -n transpr python=3.7
conda activate transpr

pip install \
    numpy \
    pyyaml \
    tensorboardX \
    plyfile\
    munch \
    scipy \
    matplotlib \
    Cython \
    PyOpenGL \
    PyOpenGL_accelerate \
    trimesh \
    huepy \
    "pillow<7" \
    tqdm \
    scikit-learn

conda install -y opencv
conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch    # change of `cudatoolkit` version might be required according to CUDA installed on the machine

# need to install separately
pip install \
    git+https://github.com/DmitryUlyanov/glumpy \
    numpy-quaternion

# pycuda
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin

./configure.py #--cuda-enable-gl
# If you receive error "nvcc: no such file or directory", add /usr/lib/cuda-<version>/bin to PATH before running this script.
python setup.py install
cd ..
