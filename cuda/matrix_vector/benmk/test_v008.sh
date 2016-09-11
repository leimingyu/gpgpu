#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v008"

./run_cuda_v008.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v008_log  > /tmp/output008  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output008

echo -e "\n"
