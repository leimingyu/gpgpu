#!/bin/bash

## dump the perf
if [ -f /tmp/benmk_cublas_log ]
then
	rm -rf /tmp/benmk_cublas_log
fi

for row in `seq 100 100 1000`;
do
	for col in `seq 100 100 1000`;
	do
		./cublas_main $row $col >> /tmp/benmk_cublas_log
	done
done


