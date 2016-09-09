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

void init1D(float *data, int len, float value)
{                                                                               
	for(int i=0; i<len; i++) {                                                 
		data[i] = value;                                        
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

void h2d_copy(float *h_data, float *d_data, int len)
{
	cudaMemcpy(d_data, h_data, sizeof(float) * len, cudaMemcpyHostToDevice);
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
// 
//----------------------------------------------------------------------------//
/*
__global__ void kernel_sgemv_v1a (const int rows,
		const int cols,
		const int col_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	//uint gx = threadIdx.x + __umul24(blockIdx.x, blockDim.x); // cols
	int gx = threadIdx.x;
	//int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x); // cols
	int gy = threadIdx.y + __mul24(blockIdx.y, blockDim.y); // rows

	int lane_id = threadIdx.x & 0x1F;

	float tmp = 0.f;
	int row_idx = gy * cols;
	for(int i=0; i<col_iters; i++)
	{
		int curr_col = gx + i * 32;
		if (curr_col < cols)
			tmp += A[row_idx + curr_col] * B[curr_col];
	}

	// warp reduction on tmp	
	tmp += __shfl_down(tmp, 16, 32);                                      
	tmp += __shfl_down(tmp,  8, 32);                                      
	tmp += __shfl_down(tmp,  4, 32);                                      
	tmp += __shfl_down(tmp,  2, 32);                                      
	tmp += __shfl_down(tmp,  1, 32);                                      

	if(lane_id == 0) {
		C[gy] = tmp;
	}
}
*/

__global__ void kernel_sgemv_v1a (const int rows,
		const int cols,
		const int col_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	//uint gx = threadIdx.x + __umul24(blockIdx.x, blockDim.x); // cols
	int gx = threadIdx.x;
	//int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x); // cols
	int gy = threadIdx.y + __mul24(blockIdx.y, blockDim.y); // rows

	int lane_id = threadIdx.x & 0x1F;

	float tmp1 = 0.f;
	float tmp2 = 0.f;
	int row_idx = gy * cols;
	for(int i=0; i<col_iters; i++)
	{
		// each warp do two warps work
		int curr_col1 = gx + i * 64;
		int curr_col2 = curr_col1 + 32;

		if (curr_col1 < cols)
			tmp1 += A[row_idx + curr_col1] * B[curr_col1];

		if (curr_col2 < cols)
			tmp2 += A[row_idx + curr_col2] * B[curr_col2];
	}

	// warp reduction on tmp	
	/*
	tmp += __shfl_down(tmp, 16, 32);
	tmp += __shfl_down(tmp,  8, 32);
	tmp += __shfl_down(tmp,  4, 32);
	tmp += __shfl_down(tmp,  2, 32);
	tmp += __shfl_down(tmp,  1, 32);
	*/
	tmp1 += __shfl_down(tmp1, 16, 32);
	tmp2 += __shfl_down(tmp2, 16, 32);

	tmp1 += __shfl_down(tmp1,  8, 32);
	tmp2 += __shfl_down(tmp2,  8, 32);

	tmp1 += __shfl_down(tmp1,  4, 32);
	tmp2 += __shfl_down(tmp2,  4, 32);

	tmp1 += __shfl_down(tmp1,  1, 32);
	tmp2 += __shfl_down(tmp2,  1, 32);

	if(lane_id == 0) {
		//C[gy] = tmp;
		C[gy] = tmp1 + tmp2;
	}
}

void test_v1a(int rows, int cols)
{
	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );
	// host
	float *A;
	float *B;
	float *C;
	checkCudaErrors(cudaMallocHost((void **)&A, 	rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&B, 	cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&C, 	rows * FLT_SIZE));
	// init
	init2D(A, rows, cols, 0.2f);
	init1D(B, cols, 0.1f);
	// dump
	//print2D(A, rows, cols);
	//print1D(B, cols);
	// device
	float *d_A;
	float *d_B;
	float *d_C;
	checkCudaErrors(cudaMalloc((void **)&d_A, 	rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_B, 	cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_C, 	rows * FLT_SIZE));
	// copy data to device
	h2d_copy(A, d_A, rows * cols);
	h2d_copy(B, d_B, cols);
	h2d_copy(C, d_C, cols);
	// start gpu timing
	cudaEventRecord(startEvent);
	//--------------------------------------------------------------------------
	// kernel
	//--------------------------------------------------------------------------
    dim3 Blk_config = dim3(32, 4, 1);                                           
    dim3 Grd_config = dim3(1, BLK(rows, 4), 1);

	kernel_sgemv_v1a <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols,32),
			d_A,
			d_B,
			d_C);
	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	//d2h_print1d(d_C, C, rows);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds << " (ms)" << endl;

	// release
	if (A != NULL)				checkCudaErrors(cudaFreeHost(A));
	if (B != NULL)				checkCudaErrors(cudaFreeHost(B));
	if (C != NULL)				checkCudaErrors(cudaFreeHost(C));

	if (d_A != NULL)			checkCudaErrors(cudaFree(d_A));
	if (d_B != NULL) 			checkCudaErrors(cudaFree(d_B));
	if (d_C != NULL)			checkCudaErrors(cudaFree(d_C));
}

int main(int argc, char **argv) {

	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);

	// warm-up
	test_v1a(100,   100);

	// 10K
	//test(100,   100);
	
	//------------------------------------------------------------------------//
	// case study 1
	//------------------------------------------------------------------------//

	// lanch a 2d grid, where x is on column with fixed warp size 32
	//test_v1a(50,   50);

	test_v1a(100,   100);
	//test_v1a(1000,   50);
	//test_v1a(100,   100);

    return(0);
}

