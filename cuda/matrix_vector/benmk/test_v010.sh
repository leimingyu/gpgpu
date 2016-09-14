#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v010"

./run_cuda_v010.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v010_log  > /tmp/output010  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output010

echo -e "\n"
