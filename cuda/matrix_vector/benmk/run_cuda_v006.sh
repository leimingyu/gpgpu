#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v006_log ]
then
	rm -rf /tmp/cuda_v006_log
fi

for row in `seq 100 100 1000`;
do
	for col in `seq 100 100 1000`;
	do
		./cuda_v006 $row $col >> /tmp/cuda_v006_log
	done
done


