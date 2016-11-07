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
		if(fabs(h_data[i]-cpu) > 1e-6){
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


__global__ void kernel_sgemv_v1a (const int rows,
		const int cols,
		const int col_iters,
		const int row_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* C)
{
	// block configure  1024 (32 x 32)

	__shared__ float B_sm[1024];
	__shared__ float C_sm[1024];

	int gx  = threadIdx.x;
	//int gx  = threadIdx.x + __mul24(blockIdx.x, blockDim.x);

	int lx = threadIdx.x & 0x1F; // lane_id
	int ly = (threadIdx.x >> 5); // warp id


	// load B to shared memory
	if(gx < cols)                                                     
		B_sm[gx] = B[gx];                                             
	else
		B_sm[gx] = 0.f; 

	__syncthreads();

	//printf("B_sm[%d] = %f\n", gx, B_sm[gx]);


	for(int j=0; j<row_iters; j++)
	{
		// index current row
		int cur_row     	= ly + j * 32;
		int start_cur_row 	= cur_row * rows;
	
		float tmp = 0.f;

		// go through  all the columns
		for(int i=0; i<col_iters; i++)
		{
			int cur_col = lx + i * 32;
			// read 32 threads at a time
			float b_reg = B_sm[cur_col];	
			tmp += A[start_cur_row + cur_col] * b_reg;
		}

		// warp reduction : summation
		// each warp  = each row
		for(int j=16; j>0; j=(j>>1)) {
			tmp += __shfl_down(tmp, j, 32);
		}

		// save row sum to shared memory: use the 1st thread in the warp
		if(lx == 0)
			C_sm[cur_row] = tmp;

		__syncthreads();
	}

	// output
	if(gx < rows)                                                     
		C[gx] = C_sm[gx];                                             
}

void test_v1a(int rows, int cols)
{
	double  flop = rows * (cols + cols) + (cols + cols);                        
	double  gflop = flop * 1e-9;

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

	// each thread on the row, do twice work load
    //dim3 Blk_config = dim3(128, 4, 1);                                           
    //dim3 Grd_config = dim3(1, BLK(rows, 4), 1);

	// 1 block = 1024 = 32 row x 32 cols
	kernel_sgemv_v1a <<< 1, 1024 >>>(rows, 
			cols, 
			BLK(cols, 32),
			BLK(rows, 32),
			d_A,
			d_B,
			d_C);

	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds << " (ms)" << endl;

	printf("GFLOPS %f \n",  gflop / (milliseconds * 1e-3));

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
}

int main(int argc, char **argv) 
{

	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);


	// 10K
	//test(100,   100);
	
	//------------------------------------------------------------------------//
	// case study 1
	//------------------------------------------------------------------------//

	// lanch a 2d grid, where x is on column with fixed warp size 32

	// warm-up
	//test_v1a(8,   128);
	//test_v1a(8,   128);

//	test_v1a(10,   20);
//	test_v1a(50,   50);
//	test_v1a(100,  500);
//	test_v1a(400,   100);
//	test_v1a(1000,   50);
	//test_v1a(400,   100);

	//test_v1a(1000,1000);

	int row = atoi(argv[1]);
	int col = atoi(argv[2]);

	test_v1a(row, col);

    return(0);
}
