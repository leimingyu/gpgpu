// Includes                                                                     
#include <stdio.h>                                                              
#include <stdlib.h>                                                              
#include <string.h>                                                             
#include <iostream>                                                             
#include <cstring>                                                              

// includes, project                                                            
#include <helper_functions.h>                                                   
#include <helper_cuda.h>                                                        

// includes, CUDA                                                               
#include <cuda.h>                                                               
#include <cudaProfiler.h>                                                       
#include <builtin_types.h>                                                      
#include <drvapi_error_string.h>   

// runtime cuda
#include <cuda_runtime.h>

using namespace std;


CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

CUresult error;

void RandomInit(float *data, int n)                                             
{                                                                               
  for (int i = 0; i < n; ++i)                                                 
  {                                                                           
	data[i] = rand() / (float)RAND_MAX;                                     
  }                                                                           
}

int main(int argc, char **argv)                                                 
{ 
  cuInit(0);
  int devID = 1;

  error = cuDeviceGet(&cuDevice, devID);
  if (error != CUDA_SUCCESS){
	std::cout<<" cuDeviceGet error! Error code: "<<error<<std::endl;
  }

  error = cuCtxCreate(&cuContext, 0, cuDevice);
  if (error != CUDA_SUCCESS){
	std::cout<<" cuCtxCreate error! Error: code: "<<error<<std::endl;
  }

  error = cuModuleLoad(&cuModule, "test.cubin");
  if (error != CUDA_SUCCESS){
	std::cout<<" cuModuleLoad error! Error code: "<<error<<std::endl;
  }

  CUfunction test1;
  error = cuModuleGetFunction(&test1, cuModule, "_Z13VecAdd_kernelPKfS0_PfPjS2_");
  if (error != CUDA_SUCCESS){
	std::cout<<" cuModuleGetFunction error! Error code: "<<error<<std::endl;
  }

  // allocate a,b,c,timer_start, timer_end 
  
  float *A = NULL; 
  float *B = NULL; 
  float *C = NULL; 
  uint *timer_start = NULL;
  uint *timer_end = NULL;

  int N = 32;
  cudaMallocManaged(&A, sizeof(float) * N); 
  cudaMallocManaged(&B, sizeof(float) * N); 
  cudaMallocManaged(&C, sizeof(float) * N); 

  cudaMallocManaged(&timer_start, sizeof(uint) * N); 
  cudaMallocManaged(&timer_end,   sizeof(uint) * N); 

  cudaDeviceSynchronize(); 


  RandomInit(A, N);
  RandomInit(B, N);
  cudaDeviceSynchronize(); 

  dim3 blksize(1,1,1);
  dim3 grdsize(1,1,1);

  uint shareMemSize = 0; 

  void *args[] = { &A, &B, &C, &timer_start, &timer_end };

  error = cuLaunchKernel(test1,  
	  grdsize.x, grdsize.y, grdsize.z,
	  blksize.x, blksize.y, blksize.z,
	  shareMemSize,
	  NULL, args, NULL);


  cudaDeviceSynchronize();                                                      
	                                                                                
  printf("time : %u %u : %u clk \n", 
	  timer_start[0], 
	  timer_end[0], 
	  timer_end[0] -  timer_start[0]); 

//  printf("time : %u %u : %u clk (%.3f clk/warp)\n", 
//  d_start[0], d_end[0], (d_end[0] - d_start[0]), ((double)(d_end[0]- d_start[0])/64.f));


  // free
  
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(timer_start);
  cudaFree(timer_end);

  return EXIT_SUCCESS;
}
