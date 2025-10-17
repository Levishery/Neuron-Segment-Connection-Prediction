#!/bin/bash

# compile pointnet if using pointnet
# cd pointnet2
# python setup.py install --user
# cd ..
# pip install torch==1.10.0 --index-url https://download.pytorch.org/whl/cu111
pip install torch==1.10.0
pip install cloud-volume==12.4.1 navis connected-components-3d cilog yacs h5py tensorboard tifffile numba plyfile opencv-python addict open3d spconv-cu117 timm torch-scatter -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install --upgrade --force-reinstall "numpy<1.23" "pandas==1.3.5" "matplotlib<3.6" -i https://pypi.mirrors.ustc.edu.cn/simple/