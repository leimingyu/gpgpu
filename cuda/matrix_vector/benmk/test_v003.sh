#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v003"

./run_cuda_v003.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v003_log  > /tmp/output003  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output003

echo -e "\n"
