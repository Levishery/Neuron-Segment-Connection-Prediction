# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))
this_dir = osp.dirname(osp.abspath(__file__))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": [
                    "-O2", 
                    "-I{}".format("{}/include".format(_ext_src_root)),
                    # "-gencode=arch=compute_61,code=sm_61",  # GTX 1080Ti
                    # "-gencode=arch=compute_70,code=sm_70",  # V100
                    # "-gencode=arch=compute_75,code=sm_75",  # T4/2080
                    # "-gencode=arch=compute_80,code=sm_80",  # A100
                    # "-gencode=arch=compute_86,code=sm_86",  # 3090
                    ],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
