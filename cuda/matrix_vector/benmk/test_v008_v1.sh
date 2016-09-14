#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v008_v1"

./run_cuda_v008_v1.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v008_log1  > /tmp/output008_v1  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w45 /tmp/output008_v1

echo -e "\n"
