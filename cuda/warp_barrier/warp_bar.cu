#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>

// warp 0: a + b = c
// warp 1: c + 3 = d
__global__  void Kernel_sync(float *a, float *b, float *c, float *d) {
  uint warpid;                                                              
  asm("mov.u32 %0,%warpid;" : "=r"(warpid));   

  int tid = threadIdx.x % 32;

  if(warpid == 0)
  {
    c[tid] = a[tid] + b[tid];
  }
  
  asm("bar.sync 0, 64;");   

  if(warpid == 1)
  {
    //printf("%f\n", c[tid]);
    d[tid] = c[tid] + 3.f; 
  }
}

__global__  void Kernel_async(float *a, float *b, float *c, float *d) {
  uint warpid;                                                              
  asm("mov.u32 %0,%warpid;" : "=r"(warpid));   

  int tid = threadIdx.x % 32;

  if(warpid == 0)
  {
    c[tid] = a[tid] + b[tid];
    asm("bar.arrive 0, 64;");   
  }
  
  //asm("bar.sync 0, 64;");   

  if(warpid == 1)
  {
    asm("bar.sync 0, 64;");   
    d[tid] = c[tid] + 3.f; 
    //asm("bar.arrive 1, 64;");   
    //printf("%f\n", c[tid]);
  }
}


int main(int argc, char **argv)
{
  if(argc != 2 ) {
    printf("Specify using sync(0)/async(1) mode.\n./warp_bar mode\n");   
    exit(1);
  }

  int mode = atoi(argv[1]); 

  // length
  const int N = 32; // two warps

  //  host a, b, c
	float *a, *b, *c, *d;
	a = (float *)malloc(sizeof(float) * N);
	b = (float *)malloc(sizeof(float) * N);
	c = (float *)malloc(sizeof(float) * N);
	d = (float *)malloc(sizeof(float) * N);

  // initialize a, b
  for(int i=0; i<N; i++) {
    a[i] = 1.f;   
    b[i] = 2.f;   
  }

  // device a, b, c
  float *d_a, *d_b, *d_c, *d_d;

	cudaMalloc(&d_a, sizeof(float) * N);
	cudaMalloc(&d_b, sizeof(float) * N);
	cudaMalloc(&d_c, sizeof(float) * N);
	cudaMalloc(&d_d, sizeof(float) * N);

	// host to device 
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // kernel: 2 warps
  if(mode == 0)
    Kernel_sync <<< 1, N * 2 >>>(d_a, d_b, d_c, d_d);
  else
    Kernel_async <<< 1, N * 2 >>>(d_a, d_b, d_c, d_d);


	cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(d, d_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for(int i=0; i<N; i++) {
    printf("[%d]: c\t%f\td\t%f\n", i, c[i], d[i]);
  }
  

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);

	free(a);
	free(b);
	free(c);
	free(d);

	cudaDeviceReset();

}
