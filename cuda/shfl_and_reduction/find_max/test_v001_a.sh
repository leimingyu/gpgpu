#!/bin/bash

##----------------------------------------------------------------------------##
echo "v001_a"
./run_v001_a.sh

awk 'BEGIN {ORS=" "}; {print $1 }' /tmp/v001_a_log  > /tmp/output_v001_a  2>&1

## 10 sampling point
## warp a line every 90 chars (8 chars for the data + 1 space)
fold -w90 /tmp/output_v001_a

echo -e "\n"
