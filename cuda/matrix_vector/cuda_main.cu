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

void init2D(float *array, int rows, int cols, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			array[i * cols + j] = value;                                        
		}                                                                       
	}                                                                           
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
__constant__ float const_mem[16000];

// Texture reference for 2D float texture                                       
texture<float, 2, cudaReadModeElementType> tex;
// Texture reference for 1D float texture                                       
texture<float, 1, cudaReadModeElementType> tex1a;
texture<float, 1, cudaReadModeElementType> tex1b;
texture<float, 1, cudaReadModeElementType> tex1c;
texture<float, 1, cudaReadModeElementType> tex1d;

//----------------------------------------------------------------------------//
// cuda kernels
//----------------------------------------------------------------------------//
__global__ void kernel_admm_nbeq_v2_part1(
		const int rows,
		const int cols,
		float* Nbeq_partial);

__global__ void kernel_admm_nbeq_v2_part2(
		const float* __restrict__ Nbeq_partial,
		const int rows,
		const int iters,
		float *Nbeq);

// part 1
__global__ void kernel_admm_nbeq_v2_part1(
		const int rows,                                                         
		const int cols,                                                         
		float* Nbeq_partial)                                                      
{                                                                               
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x); 	// cols
	uint gy = threadIdx.y + __umul24(blockDim.y, blockIdx.y);   // rows
	int lane_id = threadIdx.x & 0x1F;

	float loc_N   = 0.f;	
	float loc_beq = 0.f;	

	if(gx < rows && gy < cols)
	{
		//float loc_N    = N[gy * cols_rnd + gx]; 
		loc_N = tex2D(tex, gx, gy);
		//printf("(%d,%d) N: %f\n", gy, gx, loc_N);

		//float loc_beq  = beq[gx];
		loc_beq  = const_mem[gx];
		//printf("(%d,%d) beq : %f\n", gy, gx, loc_beq);
	}

	float data = loc_N * loc_beq;

	// use shuffle down within the warp
    for (int i=16; i>0; i>>=1 ) {                                              
        data += __shfl_down(data, i, 32);                                      
    }

	if(lane_id == 0) {
		Nbeq_partial[gy * gridDim.x + blockIdx.x] = data;
	}
}  

// part 2
__global__ void kernel_admm_nbeq_v2_part2(
		const float* __restrict__ Nbeq_partial,
		const int rows,
		const int iters,
		float *Nbeq)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	int base_ind = gx * iters;

	float tmp = 0.f;
	for(int i=0; i<iters; i++)
		tmp += Nbeq_partial[base_ind + i];

	if(gx < rows) {
		Nbeq[gx]= tmp;
		//printf("(%d) Nbeq: %f\n", gx, Nbeq[gx]);
	}
}





void test(int rows, int cols)
{
	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );

	// host
	float *N;
	float *beq;
	float *Nbeq;

	checkCudaErrors(cudaMallocHost((void **)&N,    rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&Nbeq, rows * FLT_SIZE));

	// device
	float *d_N;
	//float *d_beq;
	float *d_Nbeq;
	float *d_Nbeq_partial;

	checkCudaErrors(cudaMalloc((void **)&d_N,    rows * cols * FLT_SIZE));
	//checkCudaErrors(cudaMalloc((void **)&d_beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_Nbeq, rows * FLT_SIZE));

    // kernel configuration 
    dim3 Blk_part1 = dim3(32, 32, 1);                                           
    dim3 Grd_part1 = dim3(BLK(cols, 32), BLK(rows,32), 1);                            

    int partial_r = rows;                                                          
    int partial_c = Grd_part1.x;                                                
    cudaMallocManaged((void**)&d_Nbeq_partial , partial_r * partial_c * FLT_SIZE);

    dim3 Blk_part2 = dim3(32, 1, 1);                                            
    dim3 Grd_part2 = dim3(BLK(rows, 32), 1, 1);                                    

	// init
	init2D(N,   rows, cols, 0.2f);
	init2D(beq, 1,    cols, 0.1f);

	//timing = wtime();

	// copy to device
	//checkCudaErrors(cudaMemcpy(d_N,   N,   rows * cols * FLT_SIZE, cudaMemcpyHostToDevice));

	// copy to constant memory                                                  
    cudaMemcpyToSymbol(const_mem, beq, cols * FLT_SIZE, 0 , cudaMemcpyDeviceToDevice);

    // binding texture                                                           
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();            
    cudaArray *cuArray_N;                                                       
    checkCudaErrors(cudaMallocArray(&cuArray_N,                                 
                &floattex,                                                      
                cols,                                                              
                rows));                                                            
    checkCudaErrors(cudaMemcpyToArray(cuArray_N,                                
                0,                                                              
                0,                                                              
                N,                                                              
                rows * cols * FLT_SIZE,                                               
                cudaMemcpyDeviceToDevice));                                       
    // Bind the array to the texture                                            
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray_N));


	// start gpu timing
	cudaEventRecord(startEvent);

	// kernel
	kernel_admm_nbeq_v2_part1 <<< Grd_part1, Blk_part1 >>>(rows, 
			cols, 
			d_Nbeq_partial);

	checkCudaErrors(cudaUnbindTexture(tex));
	getLastCudaError("Kernel execution failed");
                                                                                
    kernel_admm_nbeq_v2_part2 <<< Grd_part2, Blk_part2 >>> (
            d_Nbeq_partial,
			rows,
            partial_c,
            d_Nbeq);

	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds / 1000.f << endl;

	//runtime = wtime() - timing;
	//cout << runtime << endl;

	// release
	if (N!= NULL)				checkCudaErrors(cudaFreeHost(N));
	if (beq!= NULL)				checkCudaErrors(cudaFreeHost(beq));
	if (Nbeq!= NULL)			checkCudaErrors(cudaFreeHost(Nbeq));

	if (d_N!= NULL)				checkCudaErrors(cudaFree(d_N));
	//if (d_beq!= NULL) cudaFree(d_beq);
	if (d_Nbeq!= NULL)			checkCudaErrors(cudaFree(d_Nbeq));
	if (d_Nbeq_partial!= NULL)	checkCudaErrors(cudaFree(d_Nbeq_partial));
	if (cuArray_N != NULL)		checkCudaErrors(cudaFreeArray(cuArray_N));
}


void test_v1(int rows, int cols)
{
	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );

	// host
	float *N;
	float *beq;
	float *Nbeq;

	checkCudaErrors(cudaMallocHost((void **)&N,    rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&Nbeq, rows * FLT_SIZE));

	// device
	float *d_N;
	float *d_beq;
	float *d_Nbeq;

	checkCudaErrors(cudaMalloc((void **)&d_N,    rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_Nbeq, rows * FLT_SIZE));

    // kernel configuration 
    dim3 Blk_part1 = dim3(128, 1, 1);                                           
    dim3 Grd_part1 = dim3(25, 1, 1);

	// init
	init2D(N,   rows, cols, 0.2f);
	init2D(beq, 1,    cols, 0.1f);


	// start gpu timing
	cudaEventRecord(startEvent);

	// kernel
	kernel_sgemv_v1 <<< Grd_config, Blk_config>>>(rows, 
			cols, 
			d_N,
			d_beq,
			d_Nbeq);

__global__ void kernel_sgemv_v1 (const int rows,
		const int cols,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	// 128 x 5
	__shared__ float sdata[640]; 



}

	// end of gpu timing
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds / 1000.f << endl;

	//runtime = wtime() - timing;
	//cout << runtime << endl;

	// release
	if (N!= NULL)				checkCudaErrors(cudaFreeHost(N));
	if (beq!= NULL)				checkCudaErrors(cudaFreeHost(beq));
	if (Nbeq!= NULL)			checkCudaErrors(cudaFreeHost(Nbeq));

	if (d_N!= NULL)				checkCudaErrors(cudaFree(d_N));
	if (d_beq!= NULL) 			checkCudaErrors(cudaFree(d_beq));
	if (d_Nbeq!= NULL)			checkCudaErrors(cudaFree(d_Nbeq));
}

int main(int argc, char **argv) {

	cudaDeviceProp prop;
	checkCudaErrors( cudaGetDeviceProperties(&prop, 0) );
	printf("Device: %s\n", prop.name);

	// 10K
	//test(100,   100);
	test_v1(100,   100);

	// 1K x 1K  = 1M
	//test(1000,  1000);

	// 10K x 10K = 100M
	//test(10000, 10000);

    return(0);
}

