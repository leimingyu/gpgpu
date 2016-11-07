#!/bin/bash

N="$@"

for (( i=0; i<50; i++ ))
do
	./sgemv $N $N
done
