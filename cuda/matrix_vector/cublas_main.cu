#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */

#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define FLT_SIZE sizeof(float)

#define DBG 0

using namespace std;

void init2D(float *array, int rows, int cols, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			array[i * cols + j] = value;                                        
		}                                                                       
	}                                                                           
}

void init1D(float *array, int rows, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		array[i] = value;                                        
	}                                                                           
}

void init2D_colmajor(float *array, int rows, int cols, float value)
{                                                                               
	// transpose
	// store the data in column major
	float v = value;
	for (int j = 0; j < cols; j++) { 
		for (int i = 0; i < rows; i++) {   // rows of array (column major)
			array[IDX2C(i,j,rows)] = v++; 
		} 
	}

#if DBG
	for (int i = 0; i < rows; i++) { 
		for (int j = 0; j < cols; j++) {   // rows of array (column major)
			printf("%4.f ", array[IDX2C(i,j,rows)]);
		} 
		printf("\n");
	}
#endif
}

//----------------------------------------------------------------------------//
// print functions
//----------------------------------------------------------------------------//
void printCublas_1d(int len, float *dev_ptr, float *hst_ptr)
{
	cublasStatus_t stat;
	stat = cublasGetVector(len, sizeof(float), dev_ptr, 1, hst_ptr, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { 
		fprintf(stderr, "Failed to get data (vector) from device.\n"); 
		exit(1);
	}

	for(int i=0; i<len; i++){
		printf("%5.0f ", hst_ptr[i]);
	}	printf("\n");
}


/*
	status = cublasGetVector(rows, sizeof(float), d_Nbeq, 1, Nbeq, 1);
	if (status != CUBLAS_STATUS_SUCCESS) { 
		fprintf(stderr, "Failed to get data (vector) from device.\n"); 
		exit(1);
	}

	for(int i=0; i<rows; i++){
		printf("%5.0f ", Nbeq[i]);
	}	printf("\n");
	*/


//// timer
//double timing, runtime;
//
//// seconds 
//extern double wtime(void);

inline int BLK(int number, int blksize)                                         
{                                                                               
    return (number + blksize - 1) / blksize;                                    
}                                                                               


void test(int rows, int cols)
{
	cudaEvent_t startEvent, stopEvent;
	checkCudaErrors( cudaEventCreate(&startEvent) );
	checkCudaErrors( cudaEventCreate(&stopEvent) );

	// cublas
	cublasStatus_t status;
	cublasHandle_t handle;
	//  create a cublas manager
	status = cublasCreate(&handle);                                             
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
		exit(1);
    }
	const float alpha = 1.f, beta = 0.f;

	// host
	float *N;
	float *beq;
	float *Nbeq;

	checkCudaErrors(cudaMallocHost((void **)&N,    rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMallocHost((void **)&Nbeq, rows * FLT_SIZE));

	// init
	// notes: we need to flip the row and col for cublas
	// rowmajor : 5 x 6
	// colmajor : 6 x 5
	init2D_colmajor(N,   rows, cols, 1.f);
	init1D(beq, cols, 1.f);

	// device
	float *d_N;
	float *d_beq;
	float *d_Nbeq;

	checkCudaErrors(cudaMalloc((void **)&d_N,    rows * cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_beq,  cols * FLT_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_Nbeq, rows * FLT_SIZE));

	// N : copy data to device
	cublasSetMatrix(cols, rows, sizeof(float), N, cols, d_N, cols);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write N)\n");
		exit(1);
	}

	// beq
	status = cublasSetVector(cols, sizeof(float), beq, 1, d_beq, 1);               
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! device access error (write beq)\n");
		exit(1);
	}


	//timing = wtime();
	cudaEventRecord(startEvent);

	// rows and cols are transposed by cublas
	status = cublasSgemv(handle,
			CUBLAS_OP_N, 
			rows, cols,
			&alpha, 
			d_N, rows, 
			d_beq, 1, 
			&beta, 
			d_Nbeq, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! kernel execution error.\n");
		exit(1);
	}

#if DBG
	printCublas_1d(rows, d_Nbeq, Nbeq);
#endif

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent); 

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
	cout << milliseconds / 1000.f << endl;

	//runtime = wtime() - timing;
	//cout << runtime << endl;

	// release
	if (N!= NULL)				checkCudaErrors(cudaFreeHost(N));
	if (beq!= NULL)				checkCudaErrors(cudaFreeHost(beq));
	if (Nbeq!= NULL)			checkCudaErrors(cudaFreeHost(Nbeq));

	if (d_N!= NULL)				checkCudaErrors(cudaFree(d_N));
	if (d_beq!= NULL) cudaFree(d_beq);
	if (d_Nbeq!= NULL)			checkCudaErrors(cudaFree(d_Nbeq));

	cublasDestroy(handle);
}



int main(int argc, char **argv) {

	// 10K
	//test(5,   6);
	test(100,   100);

	// 1K x 1K  = 1M
	test(1000,  1000);

	// 10K x 10K = 100M
	test(10000, 10000);

    return(0);
}

