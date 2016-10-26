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
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	// block configure  1 x 1024 (32 warps)

	// 1 x 32 (warps)
	__shared__ float smdata[32];
	__shared__ float smdata1[32];
	__shared__ float smdata2[32];
	__shared__ float smdata3[32];

	__shared__ float smdata4[32];
	__shared__ float smdata5[32];
	__shared__ float smdata6[32];
	__shared__ float smdata7[32];

	//int gx  = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	int lx  = threadIdx.x;
	int gx  = threadIdx.x;
	//int ly =  threadIdx.y;
	int gy  = threadIdx.y + __mul24(blockIdx.y, blockDim.y); // rows

	// 8x work
	//gy = (gy << 1);
	gy = (gy << 3);

	int lane_id = threadIdx.x & 0x1F;
	// warp_id = threadIdx.x / 32;
	int warp_id = (threadIdx.x >> 5);

	//int ly_offset = ly * 4;
	//int ly_offset = (ly<<2);


	float tmp  = 0.f;
	float tmp1 = 0.f;
	float tmp2 = 0.f;
	float tmp3 = 0.f;

	float tmp4 = 0.f;
	float tmp5 = 0.f;
	float tmp6 = 0.f;
	float tmp7 = 0.f;

	int row_idx  =  gy * cols;
	int row_idx1 =  row_idx + cols;
	int row_idx2 =  row_idx1 + cols;
	int row_idx3 =  row_idx2 + cols;

	int row_idx4 =  row_idx3 + cols;
	int row_idx5 =  row_idx4 + cols;
	int row_idx6 =  row_idx5 + cols;
	int row_idx7 =  row_idx6 + cols;

	for(int i=0; i<col_iters; i++)
	{
		//int curr_col  = gx + i * 1024;
		int curr_col  = gx + (i<<10);
		// work 
		if (curr_col < cols) {
			// A[gy][curr_col]
			float b = B[curr_col];
			tmp    += A[row_idx  + curr_col] * b;
			tmp1   += A[row_idx1 + curr_col] * b;
			tmp2   += A[row_idx2 + curr_col] * b;
			tmp3   += A[row_idx3 + curr_col] * b;

			tmp4   += A[row_idx4 + curr_col] * b;
			tmp5   += A[row_idx5 + curr_col] * b;
			tmp6   += A[row_idx6 + curr_col] * b;
			tmp7   += A[row_idx7 + curr_col] * b;
		}
	}

	// warp reduction on tmp	
	for(int j=16; j>0; j = (j>>1)) {
		tmp   += __shfl_down(tmp,  j, 32);                                      
		tmp1  += __shfl_down(tmp1, j, 32);                                      
		tmp2  += __shfl_down(tmp2, j, 32);                                      
		tmp3  += __shfl_down(tmp3, j, 32);                                      

		tmp4  += __shfl_down(tmp4, j, 32);                                      
		tmp5  += __shfl_down(tmp5, j, 32);                                      
		tmp6  += __shfl_down(tmp6, j, 32);                                      
		tmp7  += __shfl_down(tmp7, j, 32);                                      
	}

	// save 32 warps data to shared memory
	if(lane_id == 0) {
		smdata[warp_id] = tmp;
		smdata1[warp_id] = tmp1;
		smdata2[warp_id] = tmp2;
		smdata3[warp_id] = tmp3;

		smdata4[warp_id] = tmp4;
		smdata5[warp_id] = tmp5;
		smdata6[warp_id] = tmp6;
		smdata7[warp_id] = tmp7;
	}

	__syncthreads();

	if (warp_id == 0) {
		tmp = smdata[lx]; 
		tmp1 = smdata1[lx]; 
		tmp2 = smdata2[lx]; 
		tmp3 = smdata3[lx]; 

		tmp4 = smdata4[lx]; 
		tmp5 = smdata5[lx]; 
		tmp6 = smdata6[lx]; 
		tmp7 = smdata7[lx]; 

		for(int j=16; j>0; j = (j>>1)) {
			tmp  += __shfl_down(tmp,  j, 32);                                      
			tmp1  += __shfl_down(tmp1,  j, 32);                                      
			tmp2  += __shfl_down(tmp2,  j, 32);                                      
			tmp3  += __shfl_down(tmp3,  j, 32);                                      

			tmp4  += __shfl_down(tmp4,  j, 32);                                      
			tmp5  += __shfl_down(tmp5,  j, 32);                                      
			tmp6  += __shfl_down(tmp6,  j, 32);                                      
			tmp7  += __shfl_down(tmp7,  j, 32);                                      
		}

		if(lx == 0) {
			C[gy] = tmp;
			C[gy + 1] = tmp1;
			C[gy + 2] = tmp2;
			C[gy + 3] = tmp3;

			C[gy + 4] = tmp4;
			C[gy + 5] = tmp5;
			C[gy + 6] = tmp6;
			C[gy + 7] = tmp7;
		}
	}

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

	// x4 work
    dim3 Blk_config = dim3(1024, 1, 1);
    dim3 Grd_config = dim3(1, BLK(rows,8), 1);

	kernel_sgemv_v1a <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			BLK(cols, 1024),
			//BLK(cols,128),
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

int main(int argc, char **argv) {

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

