#ifndef KERNELS_H                                                                
#define KERNELS_H                                                                
                                                                                
#include <stdio.h>                                                              
#include <cuda.h>                                                               
#include <cuda_runtime.h>                                                       
                                                                                
#ifdef __cplusplus
extern "C" {
#endif

void run_a();
void run_b();
//__global__ void kernel_a(void); 
//__global__ void kernel_b(void); 


#ifdef __cplusplus
}
#endif
                                                                                
#endif  
