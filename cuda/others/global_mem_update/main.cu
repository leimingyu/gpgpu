#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>


__global__  void compute_alpha(const int len, float *num, float *den) 
{
	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x); 
	if (gx < len) {
		num[0] = 1.f;	
		den[0] = 2.f;	
	}
}

__global__  void update_c(const int len, const float alpha, const float *c,
		float *c_new) 
{
	int gx = threadIdx.x + __mul24(blockIdx.x, blockDim.x); 
	if (gx < len) {
		c_new[gx] = c[gx] * alpha;
	}
}

//----------------------------------------------------------------------------//
// c_new = c * (num / den);
//----------------------------------------------------------------------------//
int main(int argc, char **argv)
{
	// length
	const int N = 64;

	//  host a, b, c
	float *num, *den, *c, * c_new;

	num = (float *)malloc(sizeof(float));
	den = (float *)malloc(sizeof(float));
	c = (float *)malloc(sizeof(float) * N);
	c_new = (float *)malloc(sizeof(float) * N);

	// initialize a, b
	for(int i=0; i<N; i++) {
		c[i] = 0.3f;   
	}

	// device a, b, c
	float *d_num, *d_den, *d_c, *d_c_new;

	cudaMalloc(&d_num, sizeof(float) );
	cudaMalloc(&d_den, sizeof(float) );
	cudaMalloc(&d_c,   sizeof(float) * N);
	cudaMalloc(&d_c_new, sizeof(float) * N);

	// host to device 
	cudaMemcpy(d_c, c, sizeof(float) * N, cudaMemcpyHostToDevice);

	cudaMemset(d_num, 0, sizeof(float));
	cudaMemset(d_den, 0, sizeof(float));

	compute_alpha <<< 1, N >>>(N,d_num, d_den);

	cudaMemcpy(num, d_num, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(den, d_den, sizeof(float), cudaMemcpyDeviceToHost);

	printf("num %f \t den %f\n", num[0], den[0]);

	//update_c <<< 1, N >>> (N, d_num[0]/d_den[0], c, c_new);
	update_c <<< 1, N >>> (N, num[0]/den[0], c, c_new);

/*
	cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(d, d_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++) {
		printf("[%d]: c\t%f\td\t%f\n", i, c[i], d[i]);
	}
	*/


	cudaFree(d_den);
	cudaFree(d_num);
	cudaFree(d_c);
	cudaFree(d_c_new);

	free(num);
	free(den);
	free(c);
	free(c_new);

	cudaDeviceReset();

}
