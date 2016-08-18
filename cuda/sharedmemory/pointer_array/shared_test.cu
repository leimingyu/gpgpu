#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void kernel_test (float *my_array)
{
	// block size : 64 
	//  shared memory :    4 clusters : 4 features
	__shared__ float sdata[16];

	float *ptr[4] = {&sdata[0], &sdata[4], &sdata[8], &sdata[12]};

	uint gid = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	uint lid = threadIdx.x;

	int nclusters = 4;
	int nfeatures = 4;

	// init
	if(lid < nclusters)
	{
		for(int i=0; i<nfeatures; i++)
			sdata[lid * nfeatures + i] = lid;
	}

	__syncthreads();

	// select array pointer and print the corresponding data
	// gid & 0x00000003
	int id = gid % 4;
	float *data_ptr = ptr[id];

	printf("thread: %d: %f,%f,%f,%f\n", gid, 
			data_ptr[0], 
			data_ptr[1], 
			data_ptr[2], 
			data_ptr[3]);
}



int main(int argc, char **argv) {

  int devid=0;
  if(argc == 2)
	devid = atoi(argv[1]);

  cudaSetDevice(devid);

  int N = 64;
  float *d_a = NULL;
  cudaMallocManaged(&d_a, sizeof(float) * N); 

  for(int i = 0; i<N; i++) {
	d_a[i] = (float) i;
  }

  kernel_test <<< 1, N >>>(d_a);

  cudaDeviceSynchronize();
  cudaFree(d_a);
  cudaDeviceReset();
  return 0;
}
