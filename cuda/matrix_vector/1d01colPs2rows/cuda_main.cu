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
//  2x work per block
// 	add row_iters 
//----------------------------------------------------------------------------//
__global__ void kernel_sgemv_1d128b (const int rows,
		const int cols,
		const int col_iters,
		const int row_iters,
		const int iterwork,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	// 128 = 4 warps
	__shared__ float sb[4];
	__shared__ float sb1[4];

	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int lx = threadIdx.x;


	//int bx = blockIdx.x;	// 1 block for 1 row
	// 2x work per block
	int bx = blockIdx.x * 2;
	int bx1 = bx + 1; 


	// lx % 32
	int lane_id = threadIdx.x & 0x1F;

	// lx / 32
	int warp_id = threadIdx.x >> 5;
	
	float c  = 0.f;
	float c1 = 0.f;

	// 1st iters
	if(lx < cols) {
		c  = A[bx * cols + lx] * B[lx];
		c1 = A[bx1 * cols + lx] * B[lx];
	}

	// the rest iters
	for(int i = 1; i<col_iters; i++) {
		lx += 128; 
		if(lx < cols) {
			c  += A[bx  * cols + lx] * B[lx];
			c1 += A[bx1 * cols + lx] * B[lx];
		}
	}



	// 128 has 4 warps
	// each warp do reduction
	c  += __shfl_down(c, 16, 32);                                      
	c1 += __shfl_down(c1, 16, 32);                                      

	c += __shfl_down(c,  8, 32);                                      
	c1 += __shfl_down(c1,  8, 32);                                      

	c += __shfl_down(c,  4, 32);                                      
	c1 += __shfl_down(c1,  4, 32);                                      

	c += __shfl_down(c,  2, 32);                                      
	c1 += __shfl_down(c1,  2, 32);                                      

	c += __shfl_down(c,  1, 32);  
	c1 += __shfl_down(c1,  1, 32);  


	// 4 warps  = 4 data points
	if(lane_id == 0) {
		sb[warp_id] = c;	
		sb1[warp_id] = c1;	
	}

	__syncthreads();

	if(threadIdx.x == 0) {
		if(bx < rows) {
			C[bx]  = sb[0] + sb[1] + sb[2] + sb[3];	
		}

		if(bx1 < rows) {
			C[bx1] = sb1[0] + sb1[1] + sb1[2] + sb1[3];	
		}
	}

	// when there is more work
	for(int j=1; j<row_iters; j++)
	{
		// on gtx 970, persistent mode can run 13 x 16 = 208 blocks 
		// since I doubled the work per block, it can support 416 blocks (aka output rows)
		// if rows>416, we need to run more iterations
		//int offset = j * 416; 	
		int offset = j * iterwork; 	

		//-------------------------------------------------------------------//
		//-------------------------------------------------------------------//
		bx = blockIdx.x * 2 + offset;
		bx1 = bx + 1; 

		c  = 0.f;
		c1 = 0.f;

		// 1st iters
		if(lx < cols) {
			c  = A[bx * cols + lx] * B[lx];
			c1 = A[bx1 * cols + lx] * B[lx];
		}

		// the rest iters
		for(int i = 1; i<col_iters; i++) {
			lx += 128; 
			if(lx < cols) {
				c  += A[bx  * cols + lx] * B[lx];
				c1 += A[bx1 * cols + lx] * B[lx];
			}
		}

		// 128 has 4 warps
		// each warp do reduction
		c  += __shfl_down(c, 16, 32);                                      
		c1 += __shfl_down(c1, 16, 32);                                      

		c += __shfl_down(c,  8, 32);                                      
		c1 += __shfl_down(c1,  8, 32);                                      

		c += __shfl_down(c,  4, 32);                                      
		c1 += __shfl_down(c1,  4, 32);                                      

		c += __shfl_down(c,  2, 32);                                      
		c1 += __shfl_down(c1,  2, 32);                                      

		c += __shfl_down(c,  1, 32);  
		c1 += __shfl_down(c1,  1, 32);  


		// 4 warps  = 4 data points
		if(lane_id == 0) {
			sb[warp_id] = c;	
			sb1[warp_id] = c1;	
		}

		__syncthreads();

		if(threadIdx.x == 0) {
			if(bx < rows) {
				C[bx]  = sb[0] + sb[1] + sb[2] + sb[3];	
			}

			if(bx1 < rows) {
				C[bx1] = sb1[0] + sb1[1] + sb1[2] + sb1[3];	
			}
		}

	}
}


void test_v1a(int rows, int cols)
{
	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	int sm_num =  prop.multiProcessorCount;
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
    dim3 Blk_config = dim3(128, 1, 1);                                           
    //dim3 Grd_config = dim3(rows, 1, 1);

	int max_rows_per_iter = 16 * sm_num;
    dim3 Grd_config = dim3(max_rows_per_iter, 1, 1);	// bs 128, grd 208

	// 2x work per thread/row/block
	int iterwork = max_rows_per_iter * 2;


	//printf("iters: %d\n", BLK(cols, 4));

	kernel_sgemv_1d128b <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols, 128), // col_iter
			BLK(rows, iterwork),	// row_iter
			iterwork,
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

	// 100 x 100
	for(int i=0; i<10; i++)
		test_v1a(rows,   cols);

	test_v1a(rows,   cols);

    return(0);
}
