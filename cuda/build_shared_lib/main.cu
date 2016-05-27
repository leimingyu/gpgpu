#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include "kernels.h"

int main(int argc, char **argv)
{

    //kernel_a <<< 1, 1>>>();
    //kernel_b <<< 1, 2>>>();
	run_a();
	run_b();

	cudaDeviceReset();
}
