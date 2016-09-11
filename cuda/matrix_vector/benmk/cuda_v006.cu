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
__global__ void kernel_sgemv_v1a (const int rows,
		const int cols,
		const int col_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	int gx  = threadIdx.x;
	int gy  = threadIdx.y + __mul24(blockIdx.y, blockDim.y); // rows

	// 4x work
	gy = (gy << 2);

	int lane_id = threadIdx.x & 0x1F;

	int row_idx  = gy * cols;

	float tmp = 0.f;
	float tmp1 = 0.f;
	float tmp2 = 0.f;
	float tmp3 = 0.f;

	int stride1 = cols;
	int stride2 = (cols<<1);
	int stride3 = stride2 + cols;

	//printf("col iter : %d\n", col_iters);

	// each iteration, x4 work
	for(int i=0; i<col_iters; i++)
	{
		//int curr_col  = gx + i * 128;
		int curr_col  = gx + (i<<7);
		int curr_col1 = curr_col + 32;
		int curr_col2 = curr_col + 64;
		int curr_col3 = curr_col + 96;

		float b;
		float b1;
		float b2;
		float b3;

		int addr;
		int addr1;
		int addr2;
		int addr3;

		// prefetch 1
		if (curr_col1 < cols) 
		{
			b1    = B[curr_col1];
			addr1 = row_idx + curr_col1;
		}

		// work 
		if (curr_col < cols) 
		{
			b = B[curr_col];
			addr = row_idx + curr_col;
			tmp   += A[addr]              * b;
			tmp1  += A[addr + stride1]    * b;
			tmp2  += A[addr + stride2]    * b;
			tmp3  += A[addr + stride3]    * b;
		}

		// prefetch 2
		if (curr_col2 < cols) 
		{
			b2    = B[curr_col2];
			addr2 = row_idx + curr_col2;
		}

		// work 1
		if (curr_col1 < cols) 
		{
			tmp   += A[addr1]              * b1;
			tmp1  += A[addr1 + stride1]    * b1;
			tmp2  += A[addr1 + stride2]    * b1;
			tmp3  += A[addr1 + stride3]    * b1;
		}

		// prefetch 3
		if (curr_col3 < cols) 
		{
			b3    = B[curr_col3];
			addr3 = row_idx + curr_col3;
		}

		// work 2
		if (curr_col2 < cols) 
		{
			tmp   += A[addr2]              * b2;
			tmp1  += A[addr2 + stride1]    * b2;
			tmp2  += A[addr2 + stride2]    * b2;
			tmp3  += A[addr2 + stride3]    * b2;
		}

		// work 3
		if (curr_col3 < cols) 
		{
			tmp   += A[addr3]              * b3;
			tmp1  += A[addr3 + stride1]    * b3;
			tmp2  += A[addr3 + stride2]    * b3;
			tmp3  += A[addr3 + stride3]    * b3;
		}



		//printf("tmp %f, tmp1 %f\n", tmp, tmp1);
	}

	// warp reduction on tmp	
	tmp  += __shfl_down(tmp,  16, 32);                                      
	tmp1 += __shfl_down(tmp1, 16, 32);                                      
	tmp2 += __shfl_down(tmp2, 16, 32);                                      
	tmp3 += __shfl_down(tmp3, 16, 32);                                      

	tmp  += __shfl_down(tmp,  8, 32);                                      
	tmp1 += __shfl_down(tmp1,  8, 32);                                      
	tmp2 += __shfl_down(tmp2,  8, 32);                                      
	tmp3 += __shfl_down(tmp3,  8, 32);                                      

	tmp  += __shfl_down(tmp,  4, 32);                                      
	tmp1 += __shfl_down(tmp1,  4, 32);                                      
	tmp2 += __shfl_down(tmp2,  4, 32);                                      
	tmp3 += __shfl_down(tmp3,  4, 32);                                      

	tmp  += __shfl_down(tmp,   2, 32);                                      
	tmp1 += __shfl_down(tmp1,  2, 32);                                      
	tmp2 += __shfl_down(tmp2,  2, 32);                                      
	tmp3 += __shfl_down(tmp3,  2, 32);                                      

	tmp  += __shfl_down(tmp,   1, 32);                                      
	tmp1 += __shfl_down(tmp1,  1, 32);                                      
	tmp2 += __shfl_down(tmp2,  1, 32);                                      
	tmp3 += __shfl_down(tmp3,  1, 32);    

	if(lane_id == 0) {
		C[gy]      = tmp;
		C[gy + 1]  = tmp1;
		C[gy + 2]  = tmp2;
		C[gy + 3]  = tmp3;
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
	// kernel
	//--------------------------------------------------------------------------
    dim3 Blk_config = dim3(32, 4, 1);                                           
    dim3 Grd_config = dim3(1, BLK(rows/4, 4), 1);

	kernel_sgemv_v1a <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			//BLK(cols,32),
			//BLK(cols,64),
			BLK(cols, 128),
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

