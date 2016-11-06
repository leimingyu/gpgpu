#!/bin/bash

## dump the perf
if [ -f /tmp/cuda_v014_log ]
then
	rm -rf /tmp/cuda_v014_log
fi

for col in `seq 100 100 1000`;
do
	./cuda_v014 100 $col >> /tmp/cuda_v014_log
done
