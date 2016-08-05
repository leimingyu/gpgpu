#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel()
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	printf("cuda thread %d\n", i);
}

int main(void)
{
	// set device
	int device = 0;
	cudaSetDevice(device);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	printf("device %d : %s\n", device, prop.name);	

    cudaError_t err = cudaSuccess;

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = 1; 
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	// Create cuda events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel run time : %f ms\n" , milliseconds);


    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

