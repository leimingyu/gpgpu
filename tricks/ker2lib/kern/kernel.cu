#include <stdio.h>
#include "kernel.h"


__global__ void myk(void)
{
	printf("Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}

