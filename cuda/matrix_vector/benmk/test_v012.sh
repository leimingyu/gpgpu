#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v012"

./run_cuda_v012.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v012_log  > /tmp/output012  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output012

echo -e "\n"
