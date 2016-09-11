#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v002_log ]
then
	rm -rf /tmp/cuda_v002_log
fi

for row in `seq 100 100 1000`;
do
	for col in `seq 100 100 1000`;
	do
		./cuda_v002 $row $col >> /tmp/cuda_v002_log
	done
done


