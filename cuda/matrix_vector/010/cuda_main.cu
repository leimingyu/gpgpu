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

int check(float *d_data, float *h_data, const int rows, const int cols)
{
	float cpu = cols * 0.02;
	cudaMemcpy(h_data, d_data, sizeof(float) * rows, cudaMemcpyDeviceToHost);

	int correct = 1;
	for(int i=0; i<rows; i++) {
		//if(h_data[i] != cpu) {
		if(fabs(h_data[i] - cpu) > 1e-5) {
			fprintf(stderr, "result doesn't match! pos : %d, gpu %12.8f , cpu %12.8f\n", 
					i, h_data[i], cpu);
			correct = 0;
			break;
		}
	}
	return correct;
}

void h2d_copy(float *h_data, float *d_data, const int len)
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
// tile A: 2
//----------------------------------------------------------------------------//
__global__ void kernel_sgemv_128b (const int rows,
		const int cols,
		const int iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	// 128 x 4
	__shared__ float sb[128];

	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	//int gx1 = gx + (rows>>1);

	//int lane_id = threadIdx.x & 0x1F;
	//int warp_id = threadIdx.x >> 5;

	// load 128 data to shared memory
	sb[threadIdx.x] = B[threadIdx.x];
	__syncthreads();


	//float c[4] = {0.f, 0.f, 0.f, 0.f};
	int idx  = gx * cols;
	//int idx1 = gx1 * cols;

	float tmp  = 0.f;
	//float tmp1 = 0.f;

#pragma unroll
	//for(int i=0; i<cols; i+=4)
	//for(int i=0; i<cols; i++)
	//for(int i=0; i<cols; i+=2)
	for(int i=0; i<cols; i+=4)
	//for(int i=0; i<cols; i+=6)
	//for(int i=0; i<cols; i+=8)
//	for(int i=0; i<cols; i+=16)
	{
		int curr_addr = idx + i;
		//tmp += A[idx + i]         * B[i];
		//tmp += A[idx + i + 1]     * B[i + 1];
		//tmp += A[idx + i + 2]     * B[i + 2];
		//tmp += A[idx + i + 3]     * B[i + 3];
		
		//tmp += A[idx + i]         * sb[i];
		//tmp += A[idx + i + 1]     * sb[i + 1];
		//tmp += A[idx + i + 2]     * sb[i + 2];
		//tmp += A[idx + i + 3]     * sb[i + 3];

		tmp  += A[curr_addr]             * sb[i];
		tmp  += A[curr_addr + 1]         * sb[i + 1];
		tmp  += A[curr_addr + 2]         * sb[i + 2];
		tmp  += A[curr_addr + 3]         * sb[i + 3];
	//	tmp  += A[curr_addr + 4]         * sb[i + 4];
	//	tmp  += A[curr_addr + 5]         * sb[i + 5];
	//	tmp  += A[curr_addr + 6]         * sb[i + 6];
	//	tmp  += A[curr_addr + 7]         * sb[i + 7];

	//	tmp  += A[curr_addr + 8]         * sb[i + 8];
	//	tmp  += A[curr_addr + 9]         * sb[i + 9];
	//	tmp  += A[curr_addr + 10]        * sb[i + 10];
	//	tmp  += A[curr_addr + 11]        * sb[i + 11];
	//	tmp  += A[curr_addr + 12]        * sb[i + 12];
	//	tmp  += A[curr_addr + 13]        * sb[i + 13];
	//	tmp  += A[curr_addr + 14]        * sb[i + 14];
	//	tmp  += A[curr_addr + 15]        * sb[i + 15];
	}


	// scheme 2: with prefeching
	/*
	int idx = gx * cols;
	float tmp = 0.f;

	float preA1;
	float preA2;
	float preA3;

	for(int i=0; i<cols; i+=4)
	{
		// prefetch A1
		preA1 = A[idx + i + 1];

		// work A
		tmp += A[idx + i]         * sb[i];

		// prefetch A2
		preA2 = A[idx + i + 2];

		// work A1
		tmp += preA1         * sb[i + 1];

		// prefetch A3
		preA3 = A[idx + i + 3];

		// work A2
		tmp += preA2         * sb[i + 2];

		// work A3
		tmp += preA3         * sb[i + 3];
	}
	*/


	/*
	// tile size  = 4 
	int offset=0;

	for(int i=0; i<iters; i++)
	{
		int cur_col = i * 4;
		int loc_id = cur_col - offset;

		// tile  = 4,	blocksize = 128
		// which means, every 25 iters, load B to shared memory

		//if(cur_col % 128 == 0) {
		if( !(cur_col & 0x7F) ) {

			offset +=  128;
			//printf("cur_col : %d\n", cur_col);

			// load next 128 to shared memory
			int bid = threadIdx.x + cur_col;
			sdata[threadIdx.x] = (bid < cols)? B[bid] : 0.f;
			__syncthreads();

			//printf("sdata : %8.5f\n",sdata[threadIdx.x]);
		}


		if((cur_col + 3) < cols) {
			tmp += A[idx + cur_col]     * sdata[loc_id];
			tmp += A[idx + cur_col + 1] * sdata[loc_id + 1];
			tmp += A[idx + cur_col + 2] * sdata[loc_id + 2];
			tmp += A[idx + cur_col + 3] * sdata[loc_id + 3];
		}
		else if ((cur_col + 2) < cols) 
		{
			tmp += A[idx + cur_col]     * sdata[cur_col];
			tmp += A[idx + cur_col + 1] * sdata[cur_col + 1];
			tmp += A[idx + cur_col + 2] * sdata[cur_col + 2];
		}
		else if ((cur_col + 1) < cols) 
		{
			tmp += A[idx + cur_col]     * sdata[cur_col];
			tmp += A[idx + cur_col + 1] * sdata[cur_col + 1];
		
		}
		else if ( cur_col  < cols) 
		{
			tmp += A[idx + cur_col]     * sdata[cur_col];
		}
		else {
			// nothing
		}

	}
	*/

	if(gx < rows) {
		C[gx]  = tmp;
	}

	
	//if(gx1 < rows) {
	//	C[gx1]  = tmp1;
	//}
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
	//h2d_copy(C, d_C, rows);

	// start gpu timing
	cudaEventRecord(startEvent);
	//--------------------------------------------------------------------------
	// kernel
	//--------------------------------------------------------------------------
    //dim3 Blk_config = dim3(32, 4, 1);                                           
    //dim3 Grd_config = dim3(1, BLK(rows, 4), 1);

    dim3 Blk_config = dim3(128, 1, 1);                                           
    dim3 Grd_config = dim3(BLK(rows,128), 1, 1);
    //dim3 Grd_config = dim3(BLK(rows/2,128), 1, 1);	// x2 loads for each thread

	//printf("iters: %d\n", BLK(cols, 4));

	kernel_sgemv_128b <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols, 4),
			d_A,
			d_B,
			d_C);

	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds << " (ms)" << endl;

	//d2h_print1d(d_C, C, rows);
	if (check(d_C, C, rows, cols))	{
		printf("success!\n");
	}


	// release
	if (A != NULL)				checkCudaErrors(cudaFreeHost(A));
	if (B != NULL)				checkCudaErrors(cudaFreeHost(B));
	if (C != NULL)				checkCudaErrors(cudaFreeHost(C));

	if (d_A != NULL)			checkCudaErrors(cudaFree(d_A));
	if (d_B != NULL) 			checkCudaErrors(cudaFree(d_B));
	if (d_C != NULL)			checkCudaErrors(cudaFree(d_C));

	cudaDeviceReset();
}



int main(int argc, char **argv) {

	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);

	//test_v1a(5,   6);
	test_v1a(100,   100);
	//test_v1a(256,   512);
	//test_v1a(1000,   1000);

	
	//------------------------------------------------------------------------//
	// case study 1
	//------------------------------------------------------------------------//

	// lanch a 2d grid, where x is on column with fixed warp size 32
	//test_v1a(50,   50);

	// warm-up
	//test_v1a(100,   50);
	//test_v1a(100,   50);
	//test_v1a(1000,   50);
	//test_v1a(100,   100);

    return(0);
}
