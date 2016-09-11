#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v004"

./run_cuda_v004.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v004_log  > /tmp/output004  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output004

echo -e "\n"
