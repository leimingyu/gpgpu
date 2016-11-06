#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "cuda_main.fatbin.c"
extern void __device_stub__Z16kernel_sgemv_v1aiiiPKfS0_Pf(const int, const int, const int, const float *__restrict__, const float *__restrict__, float *__restrict__);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_17_cuda_main_cpp1_ii_8e0b8fcd(void) __attribute__((__constructor__));
void __device_stub__Z16kernel_sgemv_v1aiiiPKfS0_Pf(const int __par0, const int __par1, const int __par2, const float *__restrict__ __par3, const float *__restrict__ __par4, float *__restrict__ __par5){ const float *__T22;
 const float *__T23;
 float *__T24;
__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__cudaSetupArgSimple(__par2, 8UL);__T22 = __par3;__cudaSetupArgSimple(__T22, 16UL);__T23 = __par4;__cudaSetupArgSimple(__T23, 24UL);__T24 = __par5;__cudaSetupArgSimple(__T24, 32UL);__cudaLaunch(((char *)((void ( *)(const int, const int, const int, const float *__restrict__, const float *__restrict__, float *__restrict__))kernel_sgemv_v1a)));}
# 244 "cuda_main.cu"
void kernel_sgemv_v1a( const int __cuda_0,const int __cuda_1,const int __cuda_2,const float *__restrict__ __cuda_3,const float *__restrict__ __cuda_4,float *__restrict__ __cuda_5)
# 250 "cuda_main.cu"
{__device_stub__Z16kernel_sgemv_v1aiiiPKfS0_Pf( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 384 "cuda_main.cu"
}
# 1 "cuda_main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T210) {  __nv_dummy_param_ref(__T210); __nv_save_fatbinhandle_for_managed_rt(__T210); __cudaRegisterEntry(__T210, ((void ( *)(const int, const int, const int, const float *__restrict__, const float *__restrict__, float *__restrict__))kernel_sgemv_v1a), _Z16kernel_sgemv_v1aiiiPKfS0_Pf, (-1)); }
static void __sti____cudaRegisterAll_17_cuda_main_cpp1_ii_8e0b8fcd(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
