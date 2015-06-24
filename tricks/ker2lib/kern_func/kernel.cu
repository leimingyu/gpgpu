#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"


__global__ void myk(void)
{
	printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}

void entry(void)
{
	myk<<<1,1>>>();
	printf("CUDA status: %d\n", cudaDeviceSynchronize());
}





