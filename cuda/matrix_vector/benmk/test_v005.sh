#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v005"

./run_cuda_v005.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v005_log  > /tmp/output005  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output005

echo -e "\n"
