#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v013"

./run_cuda_v013.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v013_log  > /tmp/output013  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output013

echo -e "\n"
