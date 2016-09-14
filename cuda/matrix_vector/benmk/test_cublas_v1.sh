#!/bin/bash

##----------------------------------------------------------------------------##
echo "cublas v1"

./run_cublas_v1.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/benmk_cublas_v1_log  > /tmp/output_v1  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w45 /tmp/output_v1

echo -e "\n"
