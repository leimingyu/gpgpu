#!/bin/bash

/usr/local/cuda-7.5/bin/nvcc -ccbin g++ -I/usr/local/cuda-7.5/common/inc  -m64  --compiler-options '-fPIC' -dlink -dc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o kernel_a.o -c kernel_a.cu

/usr/local/cuda-7.5/bin/nvcc -ccbin g++ -I/usr/local/cuda-7.5/common/inc  -m64   --compiler-options '-fPIC' -dlink -dc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o kernel_b.o -c kernel_b.cu

/usr/local/cuda-7.5/bin/nvcc -ccbin g++ -shared  -m64 --compiler-options '-fPIC' -L/usr/local/cuda-7.5/lib64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o libkernels.so.1 kernel_a.o kernel_b.o -lcuda -ldl

ln -s libkernels.so.1 libkernels.so
