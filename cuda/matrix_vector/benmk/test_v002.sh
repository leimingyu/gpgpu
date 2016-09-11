#!/bin/bash

##----------------------------------------------------------------------------##
echo "cuda v002"
if [ ! -f /tmp/cuda_v002_log ]
then
	./run_cuda_v002.sh
fi

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/cuda_v002_log  > /tmp/output002  2>&1

## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output002

echo -e "\n"
