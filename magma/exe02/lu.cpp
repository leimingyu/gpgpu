#include <stdio.h>
#include <cuda.h>
#include "magma.h"
#include "magma_lapack.h"

int main (int argc, char ** argv){
	magma_init();				// initialize Magma

	real_Double_t gpu_time , cpu_time ;


	magma_int_t *piv , info ;	// piv - array of indices of inter -
	magma_int_t m = 2*8192;		// changed rows ; a,r - mxm matrices
	magma_int_t n = 100;		// b,c,c1 - mxn matrices
	magma_int_t mm=m*m;			// size of a,r
	magma_int_t mn=m*n;			// size of b,c,c1

	float *a;					// a- mxm matrix on the host
	float *r;					// r- mxm matrix on the host
	float *b;					// b- mxn matrix on the host
	float *c;					// c- mxn matrix on the host
	float *c1;					// c1 - mxn matrix on the host

	magma_int_t ione = 1;
	magma_int_t ISEED [4] = {0 ,0 ,0 ,1}; // seed

	const float alpha = 1.0; // alpha =1
	const float beta = 0.0; // beta =0
	// allocate matrices on the host
	magma_smalloc_pinned(&a , mm); // host memory for a
	magma_smalloc_pinned(&r , mm); // host memory for r
	magma_smalloc_pinned(&b , mn); // host memory for b
	magma_smalloc_pinned(&c , mn); // host memory for c
	magma_smalloc_pinned(&c1, mn); // host memory for c1

	piv = (magma_int_t *) malloc(m* sizeof (magma_int_t)); // host mem.

	// generate matrices // for piv
	lapackf77_slarnv(&ione, ISEED ,&mm ,a); // random a
	lapackf77_slaset(MagmaUpperLowerStr ,&m ,&n ,& alpha ,& alpha ,b ,&m); // b -mxn matrix of ones
	lapackf77_slacpy(MagmaUpperLowerStr ,&m ,&m,a ,&m,r ,&m); // a- >r
	printf ("upper left corner of the expected solution :\n");

	magma_sprint(4 , 4 , b, m ); // part of the expected solution

	// right hand side c=a*b
	blasf77_sgemm("N","N" ,&m ,&n ,&m ,& alpha ,a ,&m,b ,&m ,& beta ,c ,&m);
	lapackf77_slacpy(MagmaUpperLowerStr ,&m ,&n,c ,&m,c1 ,&m); //c- >c1

	// MAGMA
	// solve the linear system a*x=c, a -mxm matrix , c -mxn matrix ,
	// c is overwritten by the solution ; LU decomposition with par -
	// tial pivoting and row interchanges is used , row i of a is
	// interchanged with row piv (i)
	gpu_time = magma_wtime(); 
	magma_sgesv(m,n,a,m,piv,c,m,&info);
	gpu_time = magma_wtime() - gpu_time; 
	printf (" magma_sgesv time : %7.5f sec.\n", gpu_time); // time
	printf (" upper left corner of the magma solution :\n");

	//fixme: use magma_sgetrf_gpu()

	magma_sprint ( 4 , 4 , c, m ); // part of the Magma solution

	// LAPACK
	cpu_time = magma_wtime(); 
	lapackf77_sgesv(&m ,&n, r, &m, piv ,c1 , &m, &info);
	cpu_time = magma_wtime() - cpu_time; 
	printf("lapackf77_sgesv time : %7.5f sec.\n",cpu_time);
	printf(" upper left corner of the lapack solution :\n");
	magma_sprint( 4 , 4 , c1 , m ); // part of the Lapack solution

	magma_free_pinned(a); // free host memory
	magma_free_pinned(r); // free host memory
	magma_free_pinned(b); // free host memory
	magma_free_pinned(c); // free host memory
	magma_free_pinned(c1); // free host memory
	free(piv); // free host memory
	magma_finalize(); // finalize Magma
	return 0;
}
