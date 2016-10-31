// Docs:
// https://developer.apple.com/reference/accelerate/1513264-cblas_sgemm

extern "C" {
#include <cblas.h>
//#include <clapack.h>
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

#define DBG 0

int main(int argc, char **argv)
{
	int rows = atoi(argv[1]);
	int cols = atoi(argv[2]);

	float *A, *B, *C;
	A = (float *)malloc(rows * cols * sizeof(float));
	B = (float *)malloc(cols * sizeof(float));
	C = (float *)malloc(rows * sizeof(float));

	// init
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			A[i * cols + j] = j;
		}
	}

	for(int i=0; i<cols; i++) {
		B[i] = 1.f;
	}


	float alpha = 1.f;
	float beta = 0.f;

	// timer
	struct timeval start_event, end_event;
	struct timezone tz;

	if (gettimeofday(&start_event, &tz) == -1)                                     
		perror("Error: calling gettimeofday() not successful.\n");

	cblas_sgemv(CblasRowMajor, 
			CblasTrans,
			rows,
			cols,
			alpha,
			A,
			rows,
			B,
			1,
			beta,
			C,
			1);

	if (gettimeofday(&end_event, &tz) == -1)                                     
		perror("Error: calling gettimeofday() not successful.\n");


	double runtime_ms = ((double)(end_event.tv_sec-start_event.tv_sec) * 1000.0 +
			(double)(end_event.tv_usec-start_event.tv_usec)) / 1000.0;

	printf("runtime %f (ms)\n", runtime_ms);


#if DBG
	// print
	for(int i=0; i<rows; i++) {
		printf("%f ", C[i]);
	}
	printf("\n");
#endif

	free(A);
	free(B);
	free(C);

	return 0;
}
