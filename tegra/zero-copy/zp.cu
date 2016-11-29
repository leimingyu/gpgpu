#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <iostream>

#include <cuda_runtime.h>                                                       
#include <helper_cuda.h> 
#include <helper_functions.h>   

#define FLT_SIZE sizeof(float)
#define INT_SIZE sizeof(int)

void test();

void init_rand(float *array, int len)
{                                                                               
	for(int i=0; i<len; i++) {                                                 
		array[i] = (float)rand()/RAND_MAX;                                        
	}                                                                           
}


void print_1d(float *data, int len)
{                                                                               
	printf("\n");
	for(int i=0; i<len; i++) {                                                 
		printf("%12.6f ", data[i]);
	}                                                                           
	printf("\n");
}


inline int BLK(int number, int blksize)                                         
{                                                                               
    return (number + blksize - 1) / blksize;                                    
}                                                                               

// constant memory
//__constant__ float const_mem[16000];

__global__ void kernel(const int len,
		const float* __restrict__ A,
		int* __restrict__ const result)
{
	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if(gx < len) {
		//printf("%f\n", d_A[gx]);
		if(A[gx] > 0.5f) {
			result[gx] = 1;	
		}else {
			result[gx] = 0;	
		}
	}
}


void test()
{
	srand (time(NULL));

	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );

	int len = 10;

	//------------//
	// host
	//------------//
	float *A;
	checkCudaErrors(cudaMallocHost((void **)&A, 	len * FLT_SIZE));

	int *result;
	checkCudaErrors(cudaMallocHost((void **)&result, 	len * INT_SIZE));


	//------------//
	// device
	//------------//
	float *d_A;
	checkCudaErrors(cudaHostGetDevicePointer((void **)&d_A, (void *)A, 0));
	
	int *d_result;
	checkCudaErrors(cudaHostGetDevicePointer((void **)&d_result, (void *)result, 0));


	//------------//
	// init
	//------------//
	init_rand(A, len);
	print_1d(A, len);


	//--------------------------------------------------------------------------
	// kernel
	//--------------------------------------------------------------------------
    dim3 Blk_config = dim3(128, 1, 1);                                           
    dim3 Grd_config = dim3(BLK(len, 128), 1, 1);

	kernel<<< Grd_config, Blk_config>>>(len, d_A, d_result);

	//cudaDeviceSynchronize();

	// check result
	for(int i=0; i<len; i++)
		printf("%12d ", result[i]);
	printf("\n");

	for(int i=0; i<len; i++) {
		int value;
		if(A[i] > 0.5f) {
			value = 1;	
		} else {
			value = 0;	
		}

		if(value != result[i]) {
			fprintf(stderr, "wrong result!\n");	
			exit(0);
		}
	}

	printf("Success!\n");


	// release
	if (A != NULL)				checkCudaErrors(cudaFreeHost(A));
	if (result != NULL)				checkCudaErrors(cudaFreeHost(result));
}

int main(int argc, char **argv) {

	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);

	// Set flag to enable zero copy access
	cudaSetDeviceFlags(cudaDeviceMapHost);

	test();

    return(0);
}

