#include <stdlib.h>
#include <stdio.h>
#include "magma.h"

int main (int argc, char **argv){
    magma_init();           // initialize Magma
    magma_int_t m = 1024;   // length of a
    float *a;               // a - m- vector on the host
    float *d_a;             // d_a - m- vector a on the device

    // allocate the vector on the host
    magma_smalloc_cpu(&a , m); // host memory for a
    // allocate the vector on the device
    magma_smalloc(&d_a , m); // device memory for a
    // a={ sin (0) , sin (1) ,... , sin (m -1)}
    for(int j=0;j<m;j++) a[j] = sin((float)j);
    // copy data from host to device
    magma_ssetvector(m, a, 1 , d_a , 1 ); // copy a -> d_a
    // find the smallest index of the element of d_a with maximum
    // absolute value
    int i = magma_isamax(m, d_a, 1);
    printf("max |a[i]|: %f\n",fabs(a[i -1]));
    printf("fortran index : %d\n",i);
    free(a); // free host memory
    magma_free(d_a ); // free device memory
    magma_finalize(); // finalize Magma
    return 0;
}
// max |a[i]|: 0.999990
//
// fortran index : 700
