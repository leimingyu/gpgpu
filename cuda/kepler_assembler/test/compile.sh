#!/bin/bash

/usr/local/cuda-7.5/bin/nvcc -ccbin g++ -I/usr/local/cuda/samples/common/inc  -m64    -gencode arch=compute_30,code=compute_30 -o test.o -c test.cpp 

/usr/local/cuda-7.5/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_30,code=compute_30 -o test test.o -lcuda 
