#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v011"

./run_cuda_v011.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v011_log  > /tmp/output011  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output011

echo -e "\n"
