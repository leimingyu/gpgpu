#!/bin/bash

## dump the perf
if [ -f /tmp/v001_a_log ]
then
	rm -rf /tmp/v001_a_log
fi

for bs in 64 128 256 512 1024
do
	for da in 100 200 400 800 1000 1200 1400 1600 1800 2000
	do
		#echo "$bs $da" 
		./v001_a $bs $da >> /tmp/v001_a_log
	done
done


