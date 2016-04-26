#include <stdio.h>
#include <cuda_runtime.h>

#define MIN(X,Y) (((X)<(Y))?(X):(Y))


  __global__ void
vectorAdd(const float *A, const float *B, float *C, 
	uint *sm_id, 
	uint *blk_id, 
	uint *wp_id, 
	uint *start_clk, 
	uint *end_clk)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  uint smid, wpid, clk[2];                                                  

  asm("mov.u32 %0,%clock;" : "=r"(clk[0]));  

  C[i] = A[i] + B[i];

  asm("mov.u32 %0,%clock;" : "=r"(clk[1]));                                   

  asm("mov.u32 %0,%smid;" : "=r"(smid));                                     
  asm("mov.u32 %0,%warpid;" : "=r"(wpid)); 

  sm_id[i] = smid;
  wp_id[i] = wpid;
  blk_id[i] = blockIdx.x;
  start_clk[i] = clk[0];
  end_clk[i] = clk[1];
}

int main(void)
{
  //--------------------------------------------------------------------------//
  // set up parameters 
  //--------------------------------------------------------------------------//
  int device = 0;
  cudaSetDevice(device);

  // device info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("device %d : %s\n", device, prop.name);	
  //printf("sharedMemPerBlock: %ld\n", prop.sharedMemPerBlock);	
  //printf("regsPerBlock: %d\n", prop.regsPerBlock);	
  //printf("smx: %d\n", prop.multiProcessorCount);	

  size_t sm = prop.sharedMemPerBlock;
  int regs = prop.regsPerBlock;
  int smx = prop.multiProcessorCount;
  int blklimit = 16;

  // application info
  int app_reg = 8;
  int app_sm = 0;	
  int tmp;
  int blksize = 16;	// input

  // compute block limitations
  int sm_lmt = blklimit;
  if(app_sm > 0) {
	tmp = sm / app_sm;
	sm_lmt = (tmp < blklimit) ? tmp : blklimit;
  }

  int reg_lmt = blklimit;
  tmp = regs / (app_reg * blksize);
  reg_lmt = (tmp < blklimit) ? tmp : blklimit;

  int blocks_allocated = MIN(MIN(blklimit, sm_lmt), reg_lmt);
  printf("blocks to launch on each smx: %d\n", blocks_allocated);


  // allocate enough threads to occupy the device
  // int numElements = blocks_allocated *  blksize;
  int numElements = blocks_allocated * blksize * smx;

  int gridsize = (numElements + blksize - 1) / blksize; 

  uint *sm_id;
  uint *blk_id;
  uint *wp_id;
  uint *start_clk;
  uint *end_clk;



  cudaMallocManaged(&sm_id, sizeof(uint) * numElements);
  cudaMallocManaged(&blk_id, sizeof(uint) * numElements);
  cudaMallocManaged(&wp_id, sizeof(uint) * numElements);
  cudaMallocManaged(&start_clk, sizeof(uint) * numElements);
  cudaMallocManaged(&end_clk, sizeof(uint) * numElements);
  cudaDeviceSynchronize();

  //--------------------------------------------------------------------------//
  // run
  //--------------------------------------------------------------------------//
  cudaError_t err = cudaSuccess;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
	fprintf(stderr, "Failed to allocate host vectors!\n");
	exit(EXIT_FAILURE);
  }

  for (int i = 0; i < numElements; ++i)
  {
	h_A[i] = rand()/(float)RAND_MAX;
	h_B[i] = rand()/(float)RAND_MAX;
  }

  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  //printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  printf("CUDA kernel launch with %d blocks of %d threads\n", gridsize, blksize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  vectorAdd<<<gridsize, blksize>>>(d_A, d_B, d_C,
	  sm_id, blk_id, wp_id, start_clk, end_clk);

  cudaEventRecord(stop);

  // 
  cudaDeviceSynchronize();


  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  //printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
	fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
  }

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel run time : %f ms\n" , milliseconds);


  // print trace
  printf("thread\tsm\tblock\twarp\tstar\tend\n");
  for (int i = 0; i < numElements; ++i) {
	printf("%d\t%u\t%u\t%u\t%u\t%u\n",       
		i, sm_id[i], blk_id[i], wp_id[i], start_clk[i], end_clk[i]);
  }
  /*
	 for (int i = 0; i < numElements; ++i)
	 {
	 if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
	 {
	 fprintf(stderr, "Result verification failed at element %d!\n", i);
	 exit(EXIT_FAILURE);
	 }
	 }

	 printf("Test PASSED\n");
   */

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaFree(sm_id);
  cudaFree(blk_id);
  cudaFree(wp_id);
  cudaFree(start_clk);
  cudaFree(end_clk);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);


  cudaDeviceReset();

  // printf("Done\n");
  return 0;
}

