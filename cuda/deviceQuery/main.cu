#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>


inline
cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %sn", 
				cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}





int main(int argc, char **argv) {

	int devid;

	if(argc > 2) {
		fprintf(stderr,"too many args! specify just the device to query.\n");
		exit(1);
	} else if(argc == 2) {
		devid = atoi(argv[1]);
	} else {
		devid = 0;	
	}

	cudaDeviceProp prop;
	checkCuda( cudaGetDeviceProperties(&prop, devid) );
	printf("Device: %s\n", prop.name);

	printf("Local L1 Cache Supported  : %d\n", prop.localL1CacheSupported);
	printf("Global L1 Cache Supported : %d\n", prop.globalL1CacheSupported);

	cudaDeviceReset();

	return 0;
}
