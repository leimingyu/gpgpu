#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v007"

./run_cuda_v007.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v007_log  > /tmp/output007  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output007

echo -e "\n"
