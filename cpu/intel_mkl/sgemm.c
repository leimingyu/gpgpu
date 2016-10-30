// Docs:
// https://software.intel.com/en-us/node/429920 

//#include <cblas.h>

#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define DBG 0

int main(int argc, char **argv)
{
	int hA = atoi(argv[1]);
	int wA = atoi(argv[2]);
	int wB = atoi(argv[3]);

	int hB = wA;
	int hC = hA;
	int wC = wB;

	float *A, *B, *C;
	A = (float *)mkl_malloc(hA * wA * sizeof(float), 64);
	B = (float *)mkl_malloc(hB * wB * sizeof(float), 64);
	C = (float *)mkl_malloc(hC * wC * sizeof(float), 64);

	// init
	for(int i=0; i<hA; i++) {
		for(int j=0; j<wA; j++) {
			A[i * wA + j] = 1.f;
		}
	}

	for(int i=0; i<hB; i++) {
		for(int j=0; j<wB; j++) {
			B[i * wB + j] = j;
		}
	}


	float alpha = 1.f;
	float beta = 0.f;

	// timer
	struct timeval start_event, end_event;
	struct timezone tz;

	if (gettimeofday(&start_event, &tz) == -1)                                     
		perror("Error: calling gettimeofday() not successful.\n");

	cblas_sgemm(CblasRowMajor, 
			CblasNoTrans,
			CblasNoTrans,
			hA,
			wB,
			wA,
			alpha,
			A,
			hA,
			B,
			hB,
			beta,
			C,
			hC);

	if (gettimeofday(&end_event, &tz) == -1)                                     
		perror("Error: calling gettimeofday() not successful.\n");


	double runtime_ms = ((double)(end_event.tv_sec-start_event.tv_sec) * 1000.0 +
			(double)(end_event.tv_usec-start_event.tv_usec)) / 1000.0;

	printf("runtime %f (ms)\n", runtime_ms);


#if DBG
	// print
	for(int i=0; i<hC; i++) {
		for(int j=0; j<wC; j++) {
			printf("%f ",C[i * wC + j]);
		}
		printf("\n");
	}
#endif



	mkl_free(A);
	mkl_free(B);
	mkl_free(C);

	return 0;
}
