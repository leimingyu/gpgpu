#include <stdio.h>
#include "kernels.h"

__global__ void kernel_b(void)
{
	    printf("Kernel B: Hello from thread %d block %d\n", threadIdx.x, blockIdx.x);
}

void run_b() {
	kernel_b <<< 1, 1 >>> ();
}
