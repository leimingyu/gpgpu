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


float cpuReduction(float *h_idata, int size)
{
	int i;
	float sum =0.f;
	for(i=0; i<size; i++){
		sum += h_idata[i];	
	}
	return sum;
}

/*
static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return ((dividend % divisor) == 0) ?
           (dividend / divisor) :
           (dividend / divisor + 1);
}
*/


// parallel reduction kernel
__global__ void reduce(float *g_idata, float *g_odata, int n)
{
	extern  __shared__ float sdata[];

	int tid = threadIdx.x;
	int blockSize = blockDim.x;

	// each block threads covers two blocks data
	// we have 512 threads per block, they covers 1024 contiguous data
	uint i        = blockIdx.x * blockSize * 2 + threadIdx.x;
	// the grid size doubles accordingly
	uint gridSize = blockSize  * 2 * gridDim.x; //  512 x 2 x 16 

	//------------------------------------------------------------------------//
	// prefetching data
	//------------------------------------------------------------------------//
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	float mySum = 0.f;
	while (i < n){
		mySum += g_idata[i];
		mySum += g_idata[i+blockSize];
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512){
		if (tid < 256){
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256){
		if (tid < 128){
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128){
		if (tid <  64){
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		volatile float *smem = sdata;

		if (blockSize >=  64){
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32){
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0){
		// atomic add
		atomicAdd(&g_odata[0], sdata[0]);
	}
}


/*
__global__ void shfl_scan_test(float *data, float *g_odata, int width, float *partial_sums=NULL)
{
	extern __shared__ float sums[];

	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % warpSize;
	// determine a warp_id within a block
	int warp_id = threadIdx.x / warpSize;

	// Record "value" as a variable - we accumulate it along the way
	float value = data[id];
	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.
#pragma unroll
	for (int i=1; i<=width; i <<= 1){
		float n = __shfl_up(value, i, width);
		if (lane_id >= i) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp
	// write the sum of the warp to smem
	if (threadIdx.x % warpSize == warpSize-1){
		sums[warp_id] = value;
	}

	__syncthreads();
	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if (warp_id == 0)
	{
		float warp_sum = sums[lane_id];
		for (int i=1; i<=width; i <<= 1){
			float n = __shfl_up(warp_sum, i, width);
			if (lane_id >= i) warp_sum += n;
		}
		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	int blockSum = 0;

	if (warp_id > 0){
		blockSum = sums[warp_id-1];
	}

	value += blockSum;

	// last thread has sum, write write out the block's sum
	if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
	{
		partial_sums[blockIdx.x] = value;
	}

	// Now write out our result

	if (partial_sums == NULL && threadIdx.x == blockDim.x-1)
	{
		//partial_sums[blockIdx.x] = value;
		atomicAdd(&g_odata[0], value);
	}

}

*/

__global__ void shfl_reduction_test1 (float *g_idata, float *g_odata, size_t n)
{
	int blockSize = blockDim.x; // 1024

	uint i        = blockIdx.x * blockSize * 2 + threadIdx.x;
	uint gridSize = blockSize * 2 * gridDim.x; //  1024 x 2 x 4 

	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % warpSize;
	int warp_id = threadIdx.x / warpSize;

	__shared__ float sums[32];

	float mySum = 0.f;
	while (i < n){
		mySum += g_idata[i];
		mySum += g_idata[i+blockSize];
		i += gridSize;
	}

	__syncthreads();

	//------------------------------------------------------------------------//
	// each warp does reduction
	//------------------------------------------------------------------------//
#pragma unroll
	for (int i=1; i<=32; i <<= 1){
		float n = __shfl_up(mySum, i, 32);
		if (lane_id >= i) mySum += n;
	}

	//------------------------------------------------------------------------//
	// threads at the last lane doing summation 
	//------------------------------------------------------------------------//
	if (threadIdx.x % warpSize == warpSize-1){
		sums[warp_id] = mySum;
	}

	__syncthreads();

	//------------------------------------------------------------------------//
	// use the 1st warp to do reduction on the shared memory 
	//------------------------------------------------------------------------//
	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if (warp_id == 0)
	{
		float warp_sum = sums[lane_id];
		for (int i=1; i<=32; i <<= 1){
			float n = __shfl_up(warp_sum, i, 32);
			if (lane_id >= i) warp_sum += n;
		}
		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	float blockSum = 0;

	if (warp_id > 0){
		blockSum = sums[warp_id-1];
	}

	mySum += blockSum;


	if (threadIdx.x == blockDim.x-1){
		//partial_sums[blockIdx.x] = value;
		atomicAdd(&g_odata[0], mySum);
	}

}

__global__ void test3_kernel (float *g_idata, float *g_odata, size_t n)
{

	extern __shared__ float sums[];

	//int tid = threadIdx.x;
	int blockSize = blockDim.x; // 512

	uint i        = blockIdx.x * blockSize*2 + threadIdx.x;
	uint gridSize = blockSize * 2 * gridDim.x; //  512 x 2 x 16 

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	//int lane_id = id % warpSize;
	//int warp_id = threadIdx.x / 32;
	int lane_id = id & 0x1F;
	int warp_id = threadIdx.x >> 5;


	float mySum = 0.f;
	while (i < n){
		mySum += g_idata[i];
		mySum += g_idata[i+blockSize];
		i += gridSize;
	}

#pragma unroll
	for (int i=1; i<32; i <<=1 ) {
		mySum += __shfl_xor(mySum, i, 32);
	}


	if(lane_id == 0){
		sums[warp_id] = mySum;
	}

	__syncthreads();

	if(warp_id == 0){
		float warp_sum = sums[lane_id];
		for (int i=8; i>0; i >>=1 ) { // blocks = 16
			float n = __shfl_down(warp_sum, i, 32);
			if (lane_id < i ) warp_sum += n;
		}
		// update block sum
		if(lane_id == 0)
			atomicAdd(&g_odata[0], warp_sum);
	}

}


void test1 (float *d_idata, float *d_odata, int size, size_t bytes)
{
	printf("\ntest1:\n");

	int blockSize = 1024;

	int blks = size;
	while(blks >= blockSize){
		blks = blks/(blockSize*2);
	}

	printf("%d block size (max)\n", blockSize);
	printf("%d blks\n", blks);


	StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);

	int gridSize = blks; // 4

	float gpusum;
	int i, testIterations = 100;
	for(i=0; i< testIterations; ++i){
		gpusum = 0.f;

		cudaDeviceSynchronize();
		// reset the result
		checkCudaErrors(cudaMemset(d_odata, 0, sizeof(float)));

		sdkStartTimer(&timer);

		shfl_reduction_test1 <<< gridSize, blockSize >>>(d_idata, d_odata, size);

		cudaDeviceSynchronize();
		sdkStopTimer(&timer);
	}


	checkCudaErrors(cudaMemcpy((void*)&gpusum, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
        printf("shuffle scheme 1, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, blockSize);

	printf("GPU result = %f\n\n", gpusum);
}

__global__ void test2_kernel ( float *d_idata, float * d_odata)
{

	extern __shared__ float sdata[];

	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

	float value = d_idata[tid];

	// method 1
	// xor will exchange values between neighbour values
	for (int i=1; i<32; i*=2) {
		value += __shfl_xor(value, i, 32);
	}

	// method 2
	// use shfl_dwon
//	for (int i=16; i>0; i/=2) {
//		value += __shfl_down(value, i, 32);
//	}

	// method 3: shfl_up do the same
//	for (int i=16; i>0; i/=2) {
//		value += __shfl_up(value, i, 32);
//	}

        int lane_id = tid & 0x1F; // % 32
        int warp_id = threadIdx.x / warpSize;

	if(lane_id == 0){
		sdata[warp_id] = value;
	}

	__syncthreads();

	if(warp_id == 0){
		float warp_sum = sdata[lane_id];
		for (int i=1; i<4; i*=2) {
			float n = __shfl_down(warp_sum, i, 32);
			if (lane_id < 4/i) warp_sum += n;
		}
		// update block sum
		if(lane_id == 0)
			d_odata[blockIdx.x] = warp_sum;
	}



}

void test2()
{
	int i;
	int len = 128; // 32 x 4

	float *h_idata = (float*) malloc(sizeof(float) * len);
	float *h_odata = (float*) malloc(sizeof(float) * len);

	for(i=0; i<len/2; i++) {
		h_idata[i] = 1;
	}
	for(i=len/2; i<len; i++) {
		h_idata[i] = 2;
	}

	int smsz = sizeof(float) * 4;

	float *d_idata;
	float *d_odata;
	cudaMalloc((void**)&d_idata, sizeof(float) * len);
	cudaMalloc((void**)&d_odata, sizeof(float) * len);

	cudaMemcpy(d_idata, h_idata, sizeof(float)*len, cudaMemcpyHostToDevice);

	test2_kernel <<< 1, len, smsz >>> (d_idata, d_odata);

	cudaMemcpy(h_odata, d_odata, sizeof(float)*len, cudaMemcpyDeviceToHost);

	//print1df(h_odata, len);

	cudaFree(d_idata);
	cudaFree(d_odata);

	free(h_idata);
	free(h_odata);
}

void test3 (float *d_idata, float *d_odata, int size, size_t bytes)
{
	printf("\n\ntest2:\n");

	int blockSize = 512;

	int blks = size;
	while(blks >= blockSize){
		blks = blks/(blockSize*2); // 16
	}

	printf("%d block size (max)\n", blockSize);
	printf("%d blks\n", blks);


	StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);

	int gridSize = blks; // 4

	float gpusum;
	int i, testIterations = 100;
	for(i=0; i< testIterations; ++i){
		gpusum = 0.f;

		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemset(d_odata, 0, sizeof(float)));
		sdkStartTimer(&timer);

		test3_kernel <<< gridSize, blockSize, sizeof(float) * blks >>>(d_idata, d_odata, size);

		cudaDeviceSynchronize();
		sdkStopTimer(&timer);
	}


	checkCudaErrors(cudaMemcpy((void*)&gpusum, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
        printf("shuffle scheme 2, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, blockSize);

	printf("GPU result = %f\\nn", gpusum);
}


__global__ void test4_kernel_1 (float *in, float *partial, int nstride)
{
	uint tid = threadIdx.x + blockIdx.x * (blockDim.x * 2);
	uint stride = blockDim.x * gridDim.x * 2; // 32 x 4096 x 2
	int lane_id = threadIdx.x & 0x1F;

	float value = 0.f;

#pragma unroll
	for(int i = 0; i<nstride; i++){
		value = value + in[tid] + in[tid + blockDim.x];  
		tid = tid + stride; 
	}

	// sum up the 32 threads
	for (int i=1; i<32; i <<=1 ) {
		value += __shfl_xor(value, i, 32);
	}

	if( lane_id == 0){
		partial[blockIdx.x] = value;
	}
}

__global__ void test4_kernel_2 (float *in, float *out, int nstride)
{
	uint tid = threadIdx.x + blockIdx.x * (blockDim.x * 2);
	uint stride = blockDim.x * gridDim.x * 2; // 32 x  32 x 2
	int lane_id = threadIdx.x & 0x1F;

	float value = 0.f;

#pragma unroll
	for(int i = 0; i<nstride; i++){
		value = value + in[tid] + in[tid + blockDim.x];  
		tid = tid + stride; 
	}

	// sum up the 32 threads
	for (int i=1; i<32; i <<=1 ) {
		value += __shfl_xor(value, i, 32);
	}

	if( lane_id == 0){
		// 32 blocks atomiacally update the block sum
		//out[blockIdx.x] = value;
		atomicAdd(&out[0], value);
	}
}

void test4 (float *d_idata, float *d_odata, int size, size_t bytes)
{
	printf("\n\ntest3:\n");

	int blockSize = 32;

	int blks = size;
	while(blks >= blockSize){
		blks = blks/(blockSize*2); // 32 
		// printf("%d blks\n", blks);
	}

	printf("%d block size (max)\n", blockSize);
	printf("%d blks\n", blks);

	int nstride = size/(4096 * 32 * 2); // 64 
	int nstride1 = 4096/(32 * 32 * 2); // 2 

	float *d_partialsum;
	cudaMalloc((void**)&d_partialsum, sizeof(float) * 4096);

	StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
	float gpusum;
	int i, testIterations = 100;
	for(i=0; i< testIterations; ++i){
		gpusum = 0.f;
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemset(d_odata, 0, sizeof(float)));
		sdkStartTimer(&timer);

		test4_kernel_1 <<< 4096, 32 >>>(d_idata, d_partialsum, nstride); // 4k partial sum
		test4_kernel_2 <<< 32, 32 >>>(d_partialsum, d_odata, nstride1);

		cudaDeviceSynchronize();
		sdkStopTimer(&timer);
	}

	checkCudaErrors(cudaMemcpy((void*)&gpusum, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
        printf("shuffle scheme (test4), Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, blockSize);

	printf("GPU result = %f\n", gpusum);

	cudaFree(d_partialsum);
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


	//int size = atoi(argv[1]); // num of elements
	int size = 1 << 24;
	int maxThreads = 512;
	//int numBlocks = (size + maxThreads - 1) / maxThreads;

	//int iter = 0;
	int blks = size;
	while(blks >= maxThreads){
		//iter = iter + 1;
		blks = blks/(maxThreads*2);
	}

	//printf("%d iterations\n", iter);

	printf("%d elements\n", size);
	printf("%d threads (max)\n", maxThreads);
	printf("%d blks\n", blks);

	size_t bytes = sizeof(float) * size;
	float *h_idata = (float*) malloc(bytes);

	//------------------------------------------------------------------------//
	// init data on the host
	//------------------------------------------------------------------------//
	int i;
	for (i=0; i<size; i++){
		h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
	}

	//------------------------------------------------------------------------//
	// cpu reduction : for validation 
	//------------------------------------------------------------------------//
	float cpusum;
	cpusum = cpuReduction(h_idata, size);

	//------------------------------------------------------------------------//
	// gpu reduction 
	//------------------------------------------------------------------------//
	float gpusum;
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);

	// allocate device memory and data
	float *d_idata;
	float *d_odata;
	// device mem
	checkCudaErrors(cudaMalloc((void**)&d_idata, bytes));
	// results
	checkCudaErrors(cudaMalloc((void**)&d_odata, sizeof(float)));
	// host to device
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

	//------------------------------------------------------------------------//
	// warm-up
	//------------------------------------------------------------------------//
	dim3 dimBlock(maxThreads, 1, 1); // 512
	dim3 dimGrid(blks, 1, 1);        // 16
	int smemSize = sizeof(float) * maxThreads;

	int testIterations = 100;
	for(i=0; i< testIterations; ++i){
		gpusum = 0.f;
		checkCudaErrors(cudaMemset(d_odata, 0, sizeof(float)));

		cudaDeviceSynchronize();
		sdkStartTimer(&timer);

		reduce <<< dimGrid, dimBlock, smemSize >>> (d_idata, d_odata, size); 

		// check if kernel execution generated an error
		getLastCudaError("Kernel execution failed");

		cudaDeviceSynchronize();
		sdkStopTimer(&timer);
	}

	checkCudaErrors(cudaMemcpy((void*)&gpusum, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
        printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
               1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, maxThreads);

	printf("GPU result = %f\n", gpusum);
	printf("CPU result = %f\n", cpusum);

	//----------------------------------------------------------------------------//
	// using shuffle instructions
	//----------------------------------------------------------------------------//

	// scheme 1: load data to 32 x 32 (register matrix)
	// sum up the block and atomically add the value globally 
	test1(d_idata, d_odata, size, bytes);

	//test2();

	test3(d_idata, d_odata, size, bytes);

	// scheme2: on launch warpsize block, multiple iteration reduction
	test4(d_idata, d_odata, size, bytes);


	//----------------------------------------------------------------------------//


	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));


	free(h_idata);

	cudaDeviceReset();
}
