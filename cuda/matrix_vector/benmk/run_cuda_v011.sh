#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v011_log ]
then
	rm -rf /tmp/cuda_v011_log
fi

for row in `seq 100 100 1000`;
do
	for col in `seq 100 100 1000`;
	do
		./cuda_v011 $row $col >> /tmp/cuda_v011_log
	done
done


