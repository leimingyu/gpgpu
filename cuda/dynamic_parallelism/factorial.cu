#include <iostream>                                                             
#include <cstdio>                                                               
#include <cstdlib>                                                              
#include <helper_cuda.h>                                                        
#include <helper_string.h>  

__global__ void kern_factorial(int n, int *result)
{
	printf("result = %d\n", result[0]); 
	if(n<=1) { 
		return;
	}
	result[0] *= n;
	kern_factorial<<<1,1>>>(n-1, result);
}

int main()
{
	int max_depth = 2;
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

	int *result_d = NULL;
	cudaMallocManaged(&result_d, sizeof(int)); 

	result_d[0] = 1;
	cudaDeviceSynchronize();


	kern_factorial<<<1, 1>>>(5, result_d);


	cudaFree(result_d);

	return 0;
}
