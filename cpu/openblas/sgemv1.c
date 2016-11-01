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

#define DBG 1

int main(int argc, char **argv)
{
	int lda = 3;
	float A[] = { 1.f, 2.f,
		3.f, 4.f,
		5.f, 6.f
	};

	float B[] = {1.f, 2.f};

	float C[] = {0.f, 0.f, 0.f};

	// output :   5 , 11 , 17

	float alpha = 1.f;
	float beta = 0.f;

	// timer
	struct timeval start_event, end_event;
	struct timezone tz;

	if (gettimeofday(&start_event, &tz) == -1)                                     
		perror("Error: calling gettimeofday() not successful.\n");

	cblas_sgemv(CblasRowMajor, 
			CblasNoTrans,
			3,
			2,
			alpha,
			A,
			lda,
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
	for(int i=0; i<3; i++) {
		printf("%f ", C[i]);
	}
	printf("\n");
#endif

	return 0;
}
