#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v006"

./run_cuda_v006.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v006_log  > /tmp/output006  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output006

echo -e "\n"
