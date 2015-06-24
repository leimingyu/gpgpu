#!/bin/bash

nvcc -c kernel.cu -o kernel.o
g++ -I. -I/usr/local/cuda/include/ -I/usr/local/cuda/samples/common/inc -L/usr/local/cuda/lib64 kernel.o main.cpp -o main -lcudart
