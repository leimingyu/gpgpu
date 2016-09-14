#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v008_log1 ]
then
	rm -rf /tmp/cuda_v008_log1
fi

for row in `seq 100 50 300`;
do
	for col in `seq 100 50 300`;
	do
		./cuda_v008 $row $col >> /tmp/cuda_v008_log1
	done
done


