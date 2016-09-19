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
#include <helper_math.h>   

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
		if(h_data[i] != cpu) {
			fprintf(stderr, "result doesn't match! pos : %d, gpu %f , cpu %f\n", 
					i, h_data[i], cpu);
			correct = 0;
			break;
		}
	}
	return correct;
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
__global__ void kernel_sgemv_1d128b (const int rows,
		const int cols,
		const int col_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	__shared__ float B_sm[128];

	// 128 block thread = 4 (rows) x 32 (cols)
	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int lx = threadIdx.x;

	// assume 1 block is responsible for  4 rows	
	int bx_startrow = (blockIdx.x << 2);

	int lane_id = threadIdx.x & 0x1F;

	int warp_id = threadIdx.x >> 5;			// each warp  = one row

	// bx + warp0  = row 1 
	// bx + warp1  = row 2 
	// bx + warp2  = row 3 
	// bx + warp3  = row 4 
	int row_id = bx_startrow + warp_id;
	int row_idx = row_id * cols;

	float tmp = 0.f;

	int loc_c  = lane_id; 
	int loc_c1 = lane_id + 32;
	int loc_c2 = lane_id + 64;
	int loc_c3 = lane_id + 96;

	// each iteration  = 128 cols
	for (int i=0; i<col_iters; i++)
	{
		//int offset = i * 128;
		int col_offset = (i<<7);

		// all the 128 threads of current block load 128 data points from B to B_sm
		int col_iter = lx + i * 128; 
		if(col_iter < cols)
			B_sm[lx] = B[col_iter];
		__syncthreads();


		int col_idx  = loc_c  + col_offset;
		int col_idx1 = loc_c1 + col_offset; 
		int col_idx2 = loc_c2 + col_offset; 
		int col_idx3 = loc_c3 + col_offset; 


		if(row_id < rows)
		{
			// work on 1st 32 threads/cols
			if(col_idx < cols) {
				tmp += A[row_idx + col_idx] * B_sm[loc_c];
			}

			// work on 2nd 32 threads/cols
			if(col_idx1 < cols) {
				tmp += A[row_idx + col_idx1] * B_sm[loc_c1];
			}

			// work on 3rd 32 threads/cols
			if(col_idx2 < cols) {
				tmp += A[row_idx + col_idx2] * B_sm[loc_c2];
			}

			// work on 4th 32 threads/cols
			if(col_idx3 < cols) {
				tmp += A[row_idx + col_idx3] * B_sm[loc_c3];
			}
		}
	}

	// warp reduction
	tmp  += __shfl_down(tmp,  16, 32);                                      
	tmp  += __shfl_down(tmp,   8, 32);                                      
	tmp  += __shfl_down(tmp,   4, 32);                                      
	tmp  += __shfl_down(tmp,   2, 32);                                      
	tmp  += __shfl_down(tmp,   1, 32);                                      

	// each warp output 1 row data 
	if(lane_id == 0) {
		C[row_id] = tmp;	
	}

}


template <int CHK> void test_v1a(int rows, int cols)
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
	// kernel	: 4 x 32 per thread block
	//--------------------------------------------------------------------------
    dim3 Blk_config = dim3(128, 1, 1);                                           
	// compute how many rows to launch
	int batch_work = BLK(rows,4);
    dim3 Grd_config = dim3(batch_work, 1, 1);

	kernel_sgemv_1d128b <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols, 128), // col_iter
			d_A,
			d_B,
			d_C);



	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	if(CHK)
	{
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
		//cout << milliseconds << " (ms)" << endl;
		printf("%f (ms)\n", milliseconds);
	}

	/*
	//d2h_print1d(d_C, C, rows);
	if (check(d_C, C, rows, cols))	{
		printf("success!\n");
	}
	*/


	// release
	if (A != NULL)				checkCudaErrors(cudaFreeHost(A));
	if (B != NULL)				checkCudaErrors(cudaFreeHost(B));
	if (C != NULL)				checkCudaErrors(cudaFreeHost(C));

	if (d_A != NULL)			checkCudaErrors(cudaFree(d_A));
	if (d_B != NULL) 			checkCudaErrors(cudaFree(d_B));
	if (d_C != NULL)			checkCudaErrors(cudaFree(d_C));
}

int main(int argc, char **argv) {

	//cudaDeviceProp prop;
	//checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	//printf("Device: %s\n", prop.name);

	int rows = atoi(argv[1]);                                                   
	int cols = atoi(argv[2]);                                                   
	//printf("rows %d, cols %d\n", rows, cols);

	// 10K
	//test(100,   100);
	
	//------------------------------------------------------------------------//
	// case study 1
	//------------------------------------------------------------------------//

	// lanch a 2d grid, where x is on column with fixed warp size 32
	//test_v1a(50,   50);

	// warm-up                                                                  
	for(int i=0; i<10; i++)                                                     
		test_v1a<0>(rows,   cols);                                                  

	test_v1a<1>(rows,   cols); 



	///test_v1a(100,   50);
	//test_v1a(1000,   50);
	//test_v1a(100,   100);

    return(0);
}

