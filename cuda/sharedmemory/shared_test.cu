#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void shared_latency (float *my_array)
{
  extern __shared__ unsigned int sdata[];

  unsigned int start_time;
  unsigned int end_time;

  //int j = 1;
  float tmp1;
  float tmp2;
  float tmp3;
  float tmp4;
  float tmp5;
  float tmp6;
  float tmp7;
  float tmp8;
  float tmp9;
  float tmp10;
  float tmp11;
  float tmp12;
  float tmp13;
  float tmp14;
  float tmp15;
  float tmp16;
  float tmp17;
  float tmp18;
  float tmp19;
  float tmp20;
  float tmp21;
  float tmp22;
  float tmp23;
  float tmp24;
  float tmp25;
  float tmp26;
  float tmp27;
  float tmp28;
  float tmp29;
  float tmp30;
  float tmp31;
  float tmp32;
  float tmp33;
  //float tmp34;

  for (int i=0; i < 64; i++) {
	sdata[i] = my_array[i];
  }

  __syncthreads();

  //start_time = clock();
  asm("mov.u32 %0,%clock;" : "=r"(start_time));  

  tmp1 =sdata[0];
  tmp2 =sdata[1];
  tmp3 =sdata[2];
  tmp4 =sdata[3];
  tmp5 =sdata[4];
  tmp6 =sdata[5];
  tmp7 =sdata[6];
  tmp8 =sdata[7];
  tmp9 =sdata[8];

  tmp10=sdata[9];
  tmp11=sdata[10];
  tmp12=sdata[11];
  tmp13=sdata[12];
  tmp14=sdata[13];
  tmp15=sdata[14];
  tmp16=sdata[15];
  tmp17=sdata[16];
  tmp18=sdata[17];
  tmp19=sdata[18];

  tmp20=sdata[19];
  tmp21=sdata[20];
  tmp22=sdata[21];
  tmp23=sdata[22];
  tmp24=sdata[23];
  tmp25=sdata[24];
  tmp26=sdata[25];
  tmp27=sdata[26];
  tmp28=sdata[27];
  tmp29=sdata[28];

  tmp30=sdata[29];
  tmp31=sdata[30];
  tmp32=sdata[31];
  tmp33=sdata[32];
  //tmp34=sdata[33];


  //end_time = clock();
  asm("mov.u32 %0,%clock;" : "=r"(end_time));  

  my_array[0] = tmp1 * 0.33f;
  my_array[1] = tmp2 * 0.33f;
  my_array[2] = tmp3 * 0.33f;
  my_array[3] = tmp4 * 0.33f;
  my_array[4] = tmp5 * 0.33f;
  my_array[5] = tmp6 * 0.33f;
  my_array[6] = tmp7 * 0.33f;
  my_array[7] = tmp8 * 0.33f;
  my_array[8] = tmp9 * 0.33f;
  my_array[9] = tmp10 * 0.33f;


  my_array[10] = tmp11 * 0.33f;
  my_array[11] = tmp12 * 0.33f;
  my_array[12] = tmp13 * 0.33f;
  my_array[13] = tmp14 * 0.33f;
  my_array[14] = tmp15 * 0.33f;
  my_array[15] = tmp16 * 0.33f;
  my_array[16] = tmp17 * 0.33f;
  my_array[17] = tmp18 * 0.33f;
  my_array[18] = tmp19 * 0.33f;
  my_array[19] = tmp20 * 0.33f;


  my_array[20] = tmp21 * 0.33f;
  my_array[21] = tmp22 * 0.33f;
  my_array[22] = tmp23 * 0.33f;
  my_array[23] = tmp24 * 0.33f;
  my_array[24] = tmp25 * 0.33f;
  my_array[25] = tmp26 * 0.33f;
  my_array[26] = tmp27 * 0.33f;
  my_array[27] = tmp28 * 0.33f;
  my_array[28] = tmp29 * 0.33f;
  my_array[29] = tmp30 * 0.33f;

  my_array[30] = tmp31 * 0.33f;
  my_array[31] = tmp32 * 0.33f;
  my_array[32] = tmp33 * 0.33f;
  //my_array[33] = tmp34 * 0.33f;


  //sum_time += (end_time -start_time);
  printf("time : %u %u : %u\n", start_time, end_time, (end_time - start_time));
  //unsigned long long int time = end_time - start_time;
  //printf("time : %u\n",  time);
}



int main(int argc, char **argv) {

  int devid=0;
  if(argc == 2)
	devid = atoi(argv[1]);

  cudaSetDevice(devid);

  float *d_a = NULL;
  cudaMallocManaged(&d_a, sizeof(float) * 64); 

  for(int i = 0; i<33; i++)
  {
	d_a[i] = (float) i;
  }

  cudaDeviceSynchronize();

  size_t sharedMemSize =  sizeof(float) * 64;
  shared_latency <<< 1, 1, sharedMemSize>>>(d_a);

  cudaDeviceSynchronize();


  cudaFree(d_a);

  cudaDeviceReset();


  return 0;

}
