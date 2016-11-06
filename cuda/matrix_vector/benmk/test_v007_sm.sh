#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v007_sm"

./run_cuda_v007_sm.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v007_sm_log  > /tmp/output007_sm  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output007_sm

echo -e "\n"
