#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v007_sm_log ]
then
	rm -rf /tmp/cuda_v007_sm_log
fi

for row in `seq 100 100 1000`;
do
	for col in `seq 100 100 1000`;
	do
		./cuda_v007_sm $row $col >> /tmp/cuda_v007_sm_log
	done
done


