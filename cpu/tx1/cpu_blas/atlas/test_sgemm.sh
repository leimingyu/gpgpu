#!/bin/bash

N="$@"

for (( i=0; i<50; i++ ))
do
	./sgemm $N $N $N
done
