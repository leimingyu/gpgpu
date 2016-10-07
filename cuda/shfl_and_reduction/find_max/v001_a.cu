#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */

#include <iostream>

#include <cuda_runtime.h>                                                       
#include <helper_cuda.h> 
#include <helper_functions.h>   

#define FLT_SIZE sizeof(float)

using namespace std;

void test_v1a(int rows, int cols);

void init2D(float *array, int rows, int cols, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			array[i * cols + j] = value;                                        
		}                                                                       
	}                                                                           
}

void print2D(float *array, int rows, int cols)
{
	printf("\n");
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			printf("%5.3f ", array[i * cols + j]);
		}
		printf("\n");
	}                                                                           
}

template<typename T>
void init1D(T *data, int len, T value)
{                                                                               
	for(int i=0; i<len; i++) {                                                 
		data[i] = value;                                        
	}                                                                           
}

template<typename T>
void init1D_inc(T *data, int len)
{                                                                               
	for(int i=0; i<len; i++) {                                                 
		data[i] = (T)i;                                        
	}                                                                           
}


void print1D(float *data, int len)
{                                                                               
	printf("\n");
	for(int i=0; i<len; i++) {                                                 
		printf("%5.3f ", data[i]);
	}                                                                           
	printf("\n");
}

void d2h_print1d(float *d_data, float *h_data, const int rows)
{
	cudaMemcpy(h_data, d_data, sizeof(float) * rows, cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
		printf("%f ", h_data[i]);
	}
	printf("\n");
}

int check(float *d_data, float *h_data, const int datasize)
{
	float cpu = (float)(datasize - 1);

	cudaMemcpy(h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);

	int correct = 1;

	if(h_data[0] != cpu) {
		fprintf(stderr, "result doesn't match! gpu %f , cpu %f\n", h_data[0], cpu);
		correct = 0;
	}

	return correct;
}

template<typename T>
void h2d_copy(T *h_data, T *d_data, int len)
{
	cudaMemcpy(d_data, h_data, sizeof(T) * len, cudaMemcpyHostToDevice);
}

// timer
//double timing, runtime;
// seconds 
//extern double wtime(void);

inline int BLK(int number, int blksize)                                         
{                                                                               
    return (number + blksize - 1) / blksize;                                    
}                                                                               

// constant memory
//__constant__ float const_mem[16000];

//----------------------------------------------------------------------------//
// cuda kernel 
//----------------------------------------------------------------------------//

#define SMALL_DATA -999999

__device__ static float atomic_max(float* address, float val)
{
	// interpret the address as int
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
		old = atomicCAS( address_as_i, 
				assumed,
				__float_as_int(fmaxf(val, __int_as_float(assumed)))    // find max
				);
    } while (assumed != old);

    return __int_as_float(old);
}

//----------------------------------------------------------------------------//
// 
//----------------------------------------------------------------------------//
template <int blocksize>
__global__ void kernel_find_max(const float* __restrict__ A,
		const int warp_per_blk,
		const int datasize,
		float *A_max)
{
	extern __shared__ float sdata[];
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	int lx = threadIdx.x;
	int lane_id = threadIdx.x & 0x1F;
	int warp_id = threadIdx.x>>5;

	float tmp = (gx < datasize) ?  A[gx] : SMALL_DATA; 

	// warp reduction                                                   
	#pragma unroll                                                      
	for (int i=16; i>0; i>>=1 ) {                                       
		tmp = fmaxf(tmp, __shfl_down(tmp, i, 32) );         
	} 

	if(lane_id == 0) {
		sdata[warp_id] = tmp;
	}

	__syncthreads();

	// warp number is 17 - 32 
	if(warp_id == 0) {
		tmp = (lx < warp_per_blk) ? sdata[lx] : SMALL_DATA;

		if(blocksize == 1024) {
#pragma unroll
			for (int i=16; i>0; i>>=1 ) {                                       
				tmp = fmaxf(tmp, __shfl_down(tmp, i, 32) );
			} 
		}

		if(blocksize == 512) {
#pragma unroll
			for (int i=8; i>0; i>>=1 ) {                                       
				tmp = fmaxf(tmp, __shfl_down(tmp, i, 16) );
			} 
		}

		if(blocksize == 256) {
#pragma unroll
			for (int i=4; i>0; i>>=1 ) {                                       
				tmp = fmaxf(tmp, __shfl_down(tmp, i, 8) );
			} 
		}

		if(blocksize == 128) {
#pragma unroll
			for (int i=2; i>0; i>>=1 ) {                                       
				tmp = fmaxf(tmp, __shfl_down(tmp, i, 4) );
			} 
		}

		if(blocksize == 64) {
			if(lx == 0) {
				tmp = fmaxf(sdata[0],sdata[1]);	
			}
		}

		// check the global memory result
		// atomicMax(A_max, tmp) only works on integers
		if(lx == 0) {
			atomic_max(A_max, tmp);
		}
	}
}



template <int CHK> void test_v1(int blocksize, int datasize)
{
	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );

	// host
	float *A;
	checkCudaErrors(cudaMallocHost((void **)&A, 	datasize * FLT_SIZE));
	float *Amax;
	checkCudaErrors(cudaMallocHost((void **)&Amax, 	FLT_SIZE));

	//--------------------//
	// note
	// 	try different init scheme to verify the results
	//--------------------//
	// init 
	// assign incremental iteration to A
	init1D_inc<float>(A, datasize);

	Amax[0] = SMALL_DATA;

	// dump
	//print1D(A, datasize);
	//printf("Amax init : %f\n", Amax[0]);

	// device
	float *d_A;
	checkCudaErrors(cudaMalloc((void **)&d_A, 		datasize * FLT_SIZE));
	float *d_Amax;
	checkCudaErrors(cudaMalloc((void **)&d_Amax,	FLT_SIZE));

	// copy data to device
	h2d_copy<float>(A,    d_A,    datasize);
	h2d_copy<float>(Amax, d_Amax, 1);


	// start gpu timing
	cudaEventRecord(startEvent);
	//--------------------------------------------------------------------------
	// kernel
	//--------------------------------------------------------------------------
    dim3 Blk_config = dim3(blocksize, 1, 1);                                           
    dim3 Grd_config = dim3(BLK(datasize, blocksize), 1, 1);
	int warp_per_blk = BLK(blocksize, 32);
	size_t sm_size = warp_per_blk * FLT_SIZE; // the number of warps per block

	//-------------------------------------------//
	// note
	// 		no need for bs32, too many atomics
	//-------------------------------------------//

	if(blocksize <= 64 ) {
		//printf("use bs64 kernel\n");
		kernel_find_max<64> <<< Grd_config, Blk_config, sm_size >>>  (d_A, warp_per_blk, datasize, d_Amax);
	} else if (blocksize <= 128) {
		//printf("use bs128 kernel\n");
		kernel_find_max<128> <<< Grd_config, Blk_config, sm_size >>>  (d_A, warp_per_blk, datasize, d_Amax);
	} else if (blocksize <= 256) {
		//printf("use bs256 kernel\n");
		kernel_find_max<256> <<< Grd_config, Blk_config, sm_size >>>  (d_A, warp_per_blk, datasize, d_Amax);
	} else if (blocksize <= 512) {
		//printf("use bs512 kernel\n");
		kernel_find_max<512> <<< Grd_config, Blk_config, sm_size >>>  (d_A, warp_per_blk, datasize, d_Amax);
	} else if (blocksize <= 1024) {
		//printf("use bs1024 kernel\n");
		kernel_find_max<1024> <<< Grd_config, Blk_config, sm_size >>>  (d_A, warp_per_blk, datasize, d_Amax);
	} else { 
		fprintf(stderr, "block size is over the 1024 limit!\n");
	}

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	if(CHK)
	{
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
		printf("%f (ms)\n", milliseconds);
	}

	//// Verify the correctness
	//if (check(d_Amax, Amax, datasize))	{
	//	printf("success!\n");
	//}

	// release
	if (A   	!= NULL)			checkCudaErrors(cudaFreeHost(A));
	if (Amax 	!= NULL)			checkCudaErrors(cudaFreeHost(Amax));

	if (d_A  	!= NULL)			checkCudaErrors(cudaFree(d_A));
	if (d_Amax 	!= NULL)			checkCudaErrors(cudaFree(d_Amax));
}

int main(int argc, char **argv) {

	//cudaDeviceProp prop;
	//checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	//printf("Device: %s\n", prop.name);

	// blocksize 
	int blocksize = atoi(argv[1]);                                                   

	// datasize
	int datasize = atoi(argv[2]);                                                   

	//printf("block size %d, data size %d\n", blocksize, datasize);

	// warm-up                                                                  
	for(int i=0; i<10; i++)                                                     
		test_v1<0>(blocksize,   datasize); 

	test_v1<1>(blocksize,   datasize); 

    return(0);
}

