#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "kernel.h"


int main()
{
	myk<<<1,1>>>();
	printf("CUDA status: %d\n", cudaDeviceSynchronize());

	return 0;
}
