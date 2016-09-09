#!/bin/bash

##----------------------------------------------------------------------------##
echo "cublas"

if [ ! -f /tmp/benmk_cublas_log ]
then
	./run_cublas.sh
fi

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/benmk_cublas_log  > /tmp/output  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output

echo -e "\n"

##----------------------------------------------------------------------------##
echo "cuda v001"
if [ ! -f /tmp/cuda_v001_log ]
then
	./run_cuda_v001.sh
fi

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v001_log  > /tmp/output001  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output001

echo -e "\n"
