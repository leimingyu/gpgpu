/*
https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

*/
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define ITERS  10

inline
cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %sn", 
				cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

void run_pageable(unsigned int nElements)
{
	const unsigned int bytes = nElements * sizeof(float);
	printf("\nTransfer %u floats, size %f (MiB)\n", nElements, 
			bytes / (1024.f * 1024.f));

	float *h_a = (float*)malloc(bytes);
	float *h_b = (float*)malloc(bytes);
	float *d_a;
	cudaMalloc((int**)&d_a, bytes);

	for (int i = 0; i < nElements; ++i) h_a[i] = static_cast<float>(i);
	memset(h_b, 0, bytes);

	float time;
	cudaEvent_t startEvent, stopEvent; 
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	// host to device
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host to Device bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	// device to host
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	for (int i = 0; i < nElements; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** transfers failed ***");
			break;
		}
	}

	free(h_a);
	free(h_b);
	cudaFree(d_a);
}

void run_pinned(unsigned int nElements)
{
	const unsigned int bytes = nElements * sizeof(float);
	printf("\nTransfer %u floats, size %f (MiB)\n", nElements, 
			bytes / (1024.f * 1024.f));

	float *h_a, *h_b;
	cudaMallocHost((void**)&h_a, bytes);
	cudaMallocHost((void**)&h_b, bytes);

	float *d_a;
	cudaMalloc((int**)&d_a, bytes);

	for (int i = 0; i < nElements; ++i) h_a[i] = static_cast<float>(i);
	memset(h_b, 0, bytes);

	float time;
	cudaEvent_t startEvent, stopEvent; 
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	// host to device
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host to Device bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	// device to host
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	for (int i = 0; i < nElements; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** transfers failed ***");
			break;
		}
	}

	checkCuda(cudaDeviceSynchronize());

	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFree(d_a);
}

void run_um(unsigned int nElements)
{
	const unsigned int bytes = nElements * sizeof(float);
	printf("\nTransfer %u floats, size %f (MiB)\n", nElements, 
			bytes / (1024.f * 1024.f));

	float *h_a, *h_b;
	cudaMallocManaged((void**)&h_a, bytes);
	cudaMallocManaged((void**)&h_b, bytes);

	float *d_a;
	cudaMalloc((int**)&d_a, bytes);

	for (int i = 0; i < nElements; ++i) h_a[i] = static_cast<float>(i);
	memset(h_b, 0, bytes);

	float time;
	cudaEvent_t startEvent, stopEvent; 
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	// host to device
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host(um) to Device bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	// device to host ( um )
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Host(um) bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	for (int i = 0; i < nElements; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** transfers failed ***");
			break;
		}
	}


	// device to device ( um  )
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Device (um) bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());

	// host (um) to host (um)
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, h_a, bytes, cudaMemcpyHostToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host (um) to Host (um) bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());


	// device (um) to device (um)
	checkCuda( cudaEventRecord(startEvent, 0) );
	for (int i = 0; i < ITERS; i++)	
		checkCuda( cudaMemcpy(h_b, h_a, bytes, cudaMemcpyDeviceToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device (um) to Device (um) bandwidth (GB/s): %f\n", 
			bytes * 1e-6 * (float)(ITERS) / time);

	checkCuda(cudaDeviceSynchronize());



	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(d_a);
}

int main() {

	// output device info and transfer size
	cudaDeviceProp prop;
	checkCuda( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);

	unsigned int test_size[6] = {1e3,1e4,1e5,1e6,1e7,1e8};
	printf("test cases :\n");
	for(int i=0; i<6; i++)
		printf("%u\t",test_size[i]);
	printf("\n");
	
	printf("\n-------------\n pageable memory\n-------------\n");
	for(int i=0; i<6; i++)
		run_pageable(test_size[i]);	

	printf("\n-------------\n pinned memory\n-------------\n");
	for(int i=0; i<6; i++)
		run_pinned(test_size[i]);	

	printf("\n-------------\n unified memory\n-------------\n");
	for(int i=0; i<6; i++)
		run_um(test_size[i]);	

	cudaDeviceReset();

	return 0;
}
