__global__ void VecAdd_kernel(const float *A, const float *B, float *C, 
	uint *timer_start, 
	uint *timer_end)
{                                                                               
  int i = blockDim.x * blockIdx.x + threadIdx.x;                              
  uint start_time, end_time;

  float a = A[0];
  float b = B[0];
  float x = A[1];
  float y = B[1];
  float c; 

  start_time = clock();
  c = a + b;                                                     
  x = x + c;
  c = y + x;
  end_time = clock();

  C[i] = c;

  timer_start[0] = start_time;
  timer_end[0] = end_time;
} 
