#include <stdio.h>
#include "kernels.h"

__global__ void kernel_a(void)
{
	    printf("Kernel A: Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}

void run_a() {

		kernel_a<<<1,1>>>();
}
