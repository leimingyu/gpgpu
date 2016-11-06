#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v014"

./run_cuda_v014.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v014_log  > /tmp/output014  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output014

echo -e "\n"
