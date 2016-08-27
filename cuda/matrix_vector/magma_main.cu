#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "magma.h"
#include "magma_lapack.h"

#include <iostream>
using namespace std;

//#define FLT_SIZE sizeof(float)
//#define DBG 0


/*
void init2D(float *array, int rows, int cols, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			array[i * cols + j] = value;                                        
		}                                                                       
	}                                                                           
}
*/



void test(int rows, int cols)
{
	real_Double_t gpu_time;

	magma_int_t m = rows;
	magma_int_t n = cols;
	magma_int_t mn = m * n;

	float *a;	// mxn matrix on the host
	float *b;	// n vector on the host
	float *c, *c2;	// m vector on the host

	float *d_a, *d_b, *d_c;

	float alpha = MAGMA_S_MAKE(1.0, 0.0);	// 1
	float beta  = MAGMA_S_MAKE(1.0, 0.0);	// 1

	magma_int_t ione = 1;
	magma_int_t ISEED[4] = {0, 1, 2, 3};
	//magma_err_t err;

	magma_smalloc_pinned(&a, mn);
	magma_smalloc_pinned(&b, n);
	magma_smalloc_pinned(&c, m);
	magma_smalloc_pinned(&c2, m);

	magma_smalloc(&d_a, mn);
	magma_smalloc(&d_b, n);
	magma_smalloc(&d_c, m);

	// random matrix
	lapackf77_slarnv(&ione, ISEED, &mn, a);
	lapackf77_slarnv(&ione, ISEED, &n,  b);
	lapackf77_slarnv(&ione, ISEED, &m,  c);

	// copy memory to device
	magma_ssetmatrix(m, n, a, m, d_a, m);
	magma_ssetvector(n, b, 1, d_b, 1);
	magma_ssetvector(m, c, 1, d_c, 1);

	// start timing
	gpu_time = magma_wtime();

	// kernel
	magma_sgemv(MagmaNoTrans, m, n, alpha, d_a, m, d_b, 1, beta, d_c, 1);

	// end timing
	gpu_time = magma_wtime() - gpu_time;                                        
	printf ("magma gpu execution time : %7.5f sec.\n", gpu_time); // time


	magma_free_pinned(a);
	magma_free_pinned(b);
	magma_free_pinned(c);
	magma_free_pinned(c2);

	magma_free(d_a);
	magma_free(d_b);
	magma_free(d_c);

	magma_finalize();
}



int main(int argc, char **argv) {
	magma_init();

	// 10K
	//test(5,   6);
	test(100,   100);

	// 1K x 1K  = 1M
	test(1000,  1000);

	// 10K x 10K = 100M
	test(10000, 10000);

    return(0);
}

