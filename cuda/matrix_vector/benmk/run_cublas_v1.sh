#!/bin/bash

## dump the perf
if [ -f /tmp/benmk_cublas_v1_log ]
then
	rm -rf /tmp/benmk_cublas_v1_log
fi

for row in `seq 100 50 300`;
do
	for col in `seq 100 50 300`;
	do
		./cublas_main $row $col >> /tmp/benmk_cublas_v1_log
	done
done


