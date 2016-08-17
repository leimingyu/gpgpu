/*
	MiroBenchmark for SHFL instruction
	Reduction
*/

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

inline int BLK(int number, int blksize)
{
	return (number + blksize - 1) / blksize;
}

void print_2d_array(float* data, int rows, int cols, const char* msg)
{
	// needed for read from unified memory
	cudaDeviceSynchronize();

	if(msg != NULL) {
		printf("\n%s:\n", msg);
	}

	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			printf("%10.6f ", data[i * cols + j]);                               
		}                                                                       
		printf("\n");                                                           
	}
	printf("rows : %d \t cols : %d\n", rows, cols);
}

/*
static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
}
*/

__global__ void reduction_min_v1_part1 (float *data, float *partial)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	int lane_id = threadIdx.x & 0x1F;

	float value = data[gx];

	// sum up the 32 threads
	for (int i=16; i>0; i>>=1 ) {
		value = fminf( value, __shfl_down(value, i, 32) );
	}

	if( lane_id == 0){
		partial[blockIdx.x] = value;
	}
}

__global__ void reduction_min_v1_part2 (float *partialdata, const int cols, float *result)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	float min_data = partialdata[0];
	for(int i=1; i<cols; i++) {
		if(min_data > partialdata[i]) 
			min_data = partialdata[i];
	}
	result[0] = min_data;
}

void gpu_min_test1()
{
	printf("\n\nparallel reduction using shuffle - min op:\n\n");

	// allocation
	int N = 96;
	size_t bytes = N * sizeof(float);
	float *data;
	cudaMallocManaged((void**)&data, bytes);

	float *result;
	cudaMallocManaged((void**)&result, sizeof(float));

	// init
	for(int i=0; i<N; i++) {
		data[i] = (float) (N - i);
		//printf("%d : %f\n", i, data[i]);
	}


	dim3 Blkdim = dim3(32, 1, 1);
	dim3 Grddim = dim3(BLK(N, 32), 1, 1);

	float *partialdata;
	cudaMallocManaged((void**)&partialdata, sizeof(float) * Grddim.x);

	reduction_min_v1_part1 <<< Grddim, Blkdim >>> (data, partialdata);

	print_2d_array(partialdata, Grddim.x, 1, "partial data");

	// gather
	reduction_min_v1_part2 <<< 1, 1 >>> (partialdata, Grddim.x ,result);


	print_2d_array(result, 1, 1, "result");

	// release
	cudaFreeHost(data);
	cudaFreeHost(partialdata);
	cudaFreeHost(result);
	//cudaFree(d_partialsum);
}


__global__ void reduction_min_v2 (float *data, const int warp_num, float *result)
{
	extern __shared__ float sdata[];

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	//int lane_id = threadIdx.x & 0x1F;
	uint lane_id;
	asm("mov.u32 %0,%laneid;" : "=r"(lane_id));

	/// warning:
	///				warp id is not exact in order
	//uint warp_id;
	//asm("mov.u32 %0,%warpid;" : "=r"(warp_id));

	uint warp_id;
	warp_id = threadIdx.x >> 5;

	// printf("thread: %d, lane_id: %d, warp_id: %d\n", gx, lane_id, warp_id);

	float value = data[gx];

	// sum up the 32 threads
	for (int i=16; i>0; i>>=1 ) {
		value = fminf( value, __shfl_down(value, i, 32) );
	}

	if(lane_id == 0) {
		//partial[blockIdx.x] = value;
		sdata[warp_id] = value;
	}

	__syncthreads();

	if(gx == 0)
	{
		float min_data = sdata[0];
		
		/// notes : use define to unroll each case
		for(int i=1; i<warp_num; i++) {
			if(min_data > sdata[i]) 
				min_data = sdata[i];
		}

		result[blockIdx.x] = min_data;
	}
}


void gpu_min_test2()
{
	printf("\n\nparallel reduction using shuffle - min op:  using 1 block \n\n");

	// allocation
	int N = 96;
	size_t bytes = N * sizeof(float);
	float *data;
	cudaMallocManaged((void**)&data, bytes);

	float *result;
	cudaMallocManaged((void**)&result, sizeof(float));

	// init
	for(int i=0; i<N; i++) {
		data[i] = (float) (N - i);
		//printf("%d : %f\n", i, data[i]);
	}


	dim3 Blkdim = dim3(32 * BLK(N, 32), 1, 1);
	dim3 Grddim = dim3(1, 1, 1);

	size_t sm_size =  BLK(N, 32) * sizeof(float);
	reduction_min_v2 <<< Grddim, Blkdim, sm_size >>> (data, BLK(N, 32), result);

	print_2d_array(result, 1, 1, "result");

	// release
	cudaFreeHost(data);
	cudaFreeHost(result);
}


int main(int argc, char *argv[])
{
	int cuda_device = 0;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	cuda_device = findCudaDevice(argc, (const char **)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDevice(&cuda_device));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
	printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n\n",
			deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);


	// __shfl intrinsic needs SM 3.0 or higher
	if (deviceProp.major < 3)
	{
		printf("> __shfl() intrinsic requires device SM 3.0+\n");
		printf("> Waiving test.\n");
		exit(EXIT_SUCCESS);
	}


	gpu_min_test1();


	gpu_min_test2();



	cudaDeviceReset();
}
