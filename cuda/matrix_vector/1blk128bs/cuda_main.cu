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

//----------------------------------------------------------------------------//
// description
// 	each block work on 4 rows
//----------------------------------------------------------------------------//
__global__ void kernel_sgemv_1b128bs (const int rows,
		const int cols,
		const int col_iters,
		const int row_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	__shared__ float B_sm[128];
	__shared__ float out_sm[128];

	// 128 block thread = 4 (rows) x 32 (cols)
	//int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int lx = threadIdx.x;
	int lane_id = threadIdx.x & 0x1F;
	int warp_id = threadIdx.x >> 5;			// each warp  = one row

	//printf("col_iters = %d \n", col_iters);
	int loc_c  = lane_id; 
	int loc_c1 = lane_id + 32;
	int loc_c2 = lane_id + 64;
	int loc_c3 = lane_id + 96;


	for(int i=0; i<row_iters; i++)
	{
		//int row_offset = i * 4; 
		int row_offset = (i<<2); 
		// row_offset + warp i
		int row_id = row_offset + warp_id;
		int row_idx = row_id * cols;
	
		//--------------------------------------//
		// each column iteration  = 128 cols
		//--------------------------------------//
		float tmp = 0.f;
		for (int j=0; j<col_iters; j++)
		{
			//int offset = j * 128;
			int col_offset = (j<<7);
			// all the 128 threads of current block load 128 data points from B to B_sm
			int col_iter = lx + col_offset; 
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
				// work on 2nd 32 threads/cols
				// work on 3rd 32 threads/cols
				// work on 4th 32 threads/cols
				if(col_idx  < cols) tmp += A[row_idx + col_idx]  * B_sm[loc_c];
				if(col_idx1 < cols) tmp += A[row_idx + col_idx1] * B_sm[loc_c1];
				if(col_idx2 < cols) tmp += A[row_idx + col_idx2] * B_sm[loc_c2];
				if(col_idx3 < cols) tmp += A[row_idx + col_idx3] * B_sm[loc_c3];
			}

		} // end column iteration

		// warp reduction
		tmp  += __shfl_down(tmp,  16, 32);                                      
		tmp  += __shfl_down(tmp,   8, 32);                                      
		tmp  += __shfl_down(tmp,   4, 32);                                      
		tmp  += __shfl_down(tmp,   2, 32);                                      
		tmp  += __shfl_down(tmp,   1, 32);                                      

		// each warp output 1 row data 
		if(lane_id == 0) {
			out_sm[row_id] = tmp;	
		}
	} // end row iterations

	// output
	if(lx < rows) {
		C[lx] = out_sm[lx];
	}
}


void test_v1a(int rows, int cols)
{
	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	//int sm_num =  prop.multiProcessorCount;
	//printf("sm : %d\n", sm_num);

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
	// 	each block for one row of A
	//--------------------------------------------------------------------------

	// bs 128, max 16 blks per sm
	// gtx 970 has 13 smx
	// current launch 1 block : 128 threads
    dim3 Blk_config = dim3(128, 1, 1);                                           

	// compute how many rows to launch
	//int batch_work = BLK(rows,4);
    //dim3 Grd_config = dim3(batch_work, 1, 1);
    dim3 Grd_config = dim3(1, 1, 1);


	//printf("iters: %d\n", BLK(cols, 4));

	kernel_sgemv_1b128bs <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols, 128), // col_iter
			BLK(rows, 4),   // row_iter
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

	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);

	// 32 x 100
//	for(int i=0; i<10; i++)
//		test_v1a(rows,   cols);
//
	test_v1a(rows,   cols);

    return(0);
}
