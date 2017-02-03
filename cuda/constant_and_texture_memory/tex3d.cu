#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

inline int BLK(int number, int blksize)                                         
{                                                                               
	return (number + blksize - 1) / blksize;                                    
}    

// 1024 * 4 = 4k
__constant__ float beq_cnstmem[1024];


// Texture reference for 2D float texture                                       
texture<float, 2, cudaReadModeElementType> tex_N;


#define FLT_SIZE sizeof(float)
// part 1
__global__ void kernel_v1_part1(
		//const float* __restrict__ N,
		//const float* __restrict__ beq,
		const int rows_rnd,                                                         
		const int cols_rnd,                                                         
		float* Nbeq_partial)                                                      
{                                                                               
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x); 	// cols
	uint gy = threadIdx.y + __umul24(blockDim.y, blockIdx.y);  // rows

	// threadIdx.x % 32
	int lane_id = threadIdx.x & 0x1F;

	// load to reg
	//float loc_N    = N[gy * cols_rnd + gx]; 
	float loc_N = tex2D(tex_N, gx, gy);

	//float loc_beq  = beq[gx];
	float loc_beq  = beq_cnstmem[gx];

	float data = loc_N * loc_beq;

	// use shuffle down within the warp
    for (int i=16; i>0; i>>=1 ) {                                              
        data += __shfl_down(data, i, 32);                                      
    }

	if(lane_id == 0) {
		Nbeq_partial[gy * gridDim.x + blockIdx.x] = data;
	}
}  

// part 2
__global__ void kernel_v1_part2(
		const float* __restrict__ Nbeq_partial,
		const int cols,
		float *Nbeq)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	int base_ind = gx * cols;

	float tmp = 0.f;
	for(int i=0; i<cols; i++)
		tmp += Nbeq_partial[base_ind + i];

	Nbeq[gx]= tmp;
}

//----------------------------------------------------------------------------//
// kernel v2
//----------------------------------------------------------------------------//
__global__ void kernel_v2 (float *N, 
		float *beq, 
		const int c, 
		const int iters,
		float *Nbeq)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x); 	// cols
	uint gy = threadIdx.y + __umul24(blockDim.y, blockIdx.y);   // rows

	// threadIdx.x % 32
	int lane_id = threadIdx.x & 0x1F;

	float data = 0.f;
	for(int i=0; i<iters; i++)
	{
		data += N[gy * c + (gx + i * 32)] * beq[gx + i * 32];
	}

	// use shuffle down within the warp
	for (int i=16; i>0; i>>=1 ) {                                              
		data += __shfl_down(data, i, 32);                                      
	}

	if(lane_id == 0) {
		Nbeq[gy] = data;
	}
}

//----------------------------------------------------------------------------//
// kernel v3
//----------------------------------------------------------------------------//
__global__ void kernel_v3_part1(const float* __restrict__ N,                                
		const float* __restrict__ beq,                                                       
		const int rows_rnd,                                                         
		const int cols_rnd,                                                         
		float* Nbeq_partial)                                                      
{                                                                               
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x); 	// cols
	uint gy = threadIdx.y + __umul24(blockDim.y, blockIdx.y);  // rows

	// threadIdx.x % 16 
	int lane_id = threadIdx.x & 0x0F;

	// load to reg
	float loc_N    = N[gy * cols_rnd + gx]; 
	float loc_beq  = beq[gx];

	float data = loc_N * loc_beq;

	// use shuffle down within the warp
    for (int i=8; i>0; i>>=1 ) {                                              
        data += __shfl_down(data, i, 16);                                      
    }

	if(lane_id == 0) {
		Nbeq_partial[gy * gridDim.x + blockIdx.x] = data;
	}
}  

// part 2
__global__ void kernel_v3_part2(
		const float* __restrict__ Nbeq_partial,
		const int cols,
		float *Nbeq)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	int base_ind = gx * cols;

	float tmp = 0.f;
	for(int i=0; i<cols; i++)
		tmp += Nbeq_partial[base_ind + i];

	Nbeq[gx]= tmp;
}

void run_ver1();
void run_ver2();
void run_ver3();


//----------------------------------------------------------------------------//
// main
//----------------------------------------------------------------------------//
int main(void)
{

	run_ver1();
	//run_ver2();
	//run_ver3();

	cudaDeviceReset();
	return 0;
}

void run_ver1()
{
	float *N;                                                                   
    float *beq;                                                                 
    float *Nbeq_partial;                                                        
    float *Nbeq;                                                                
                                                                                
    int rows = 1000;                                                            
    int cols = 1000;                                                            
                                                                                
    int r = (rows == 1) ? 1 : BLK(rows, 32) * 32;                               
    int c = (cols == 1) ? 1 : BLK(cols, 32) * 32;                               
                                                                                
    // x cols, y rows                                                           
    dim3 Blk_part1 = dim3(32, 32, 1);                                           
    dim3 Grd_part1 = dim3(BLK(c, 32), BLK(r,32), 1);                            
                                                                                
    dim3 Blk_part2 = dim3(32, 1, 1);                                            
    dim3 Grd_part2 = dim3(BLK(r, 32), 1, 1);                                    
                                                                                
    int partial_r = r;                                                          
    int partial_c = Grd_part1.x;                                                
                                                                                
    //cudaMallocManaged((void**)&N,   r * c * FLT_SIZE);                          
	N = (float*) malloc(r * c * FLT_SIZE);
    cudaMallocManaged((void**)&beq, c * FLT_SIZE);                              
    cudaMallocManaged((void**)&Nbeq, c * FLT_SIZE);                             
    cudaMallocManaged((void**)&Nbeq_partial , partial_r * partial_c * FLT_SIZE);


	for(int i=0; i<r; i++) {
		for(int j=0; j<c; j++) {
			N[i * c + j] = 1.f;
		}	
	}

	for(int j=0; j<c; j++) {
		beq[j] = 3.f;
	}	

	// copy to constant memory
	cudaMemcpyToSymbol(beq_cnstmem, beq, c * FLT_SIZE, 0 , cudaMemcpyHostToDevice);

	//---------------------------//
	// bining texture
	//---------------------------//
	//checkCudaErrors(cudaBindTexture2D(NULL, N_tex, N, c, r, c* sizeof(float)));
	    //prepare channel format descriptor for passing texture into kernels
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();
	cudaArray *cuArray_N;
	// width = col, hight = row
	checkCudaErrors(cudaMallocArray(&cuArray_N,                                   
				&floattex,                               
				c,                                      
				r));

	checkCudaErrors(cudaMemcpyToArray(cuArray_N,
				0,                                        
				0,                                        
				N,                                    
				r * c * FLT_SIZE,                                     
				cudaMemcpyHostToDevice)); 

	// Bind the array to the texture                                            
    checkCudaErrors(cudaBindTextureToArray(tex_N, cuArray_N));



    // gpu timer                                                                
    float gpuTime;                                                              
    cudaEvent_t startEvent, stopEvent;                                          
    checkCudaErrors( cudaEventCreate(&startEvent) );                            
    checkCudaErrors( cudaEventCreate(&stopEvent) );                             
    cudaEventRecord(startEvent);                                                
                                                                                
	//------------------------------------------------------------------------//
	// version 1
	//------------------------------------------------------------------------//
    kernel_v1_part1 <<< Grd_part1, Blk_part1 >>> (
			//N,                  
            //beq,                                                                
            r,                                                                  
            c,                                                                  
            Nbeq_partial);                                                      
                                                                                
                                                                                
    kernel_v1_part2 <<< Grd_part2, Blk_part2 >>> (                    
            Nbeq_partial,                                                       
            partial_c,                                                          
            Nbeq);                                                              
                                                                                
                                                                                
    cudaEventRecord(stopEvent);                                                 
    cudaEventSynchronize(stopEvent);                                            
    cudaEventElapsedTime(&gpuTime,startEvent,stopEvent);                        
    printf("=>\t\t\tGPU execution time : %f ms\n", gpuTime);                    
                                                                                
                                                                                
    cudaDeviceSynchronize();                                                    

	checkCudaErrors(cudaUnbindTexture(tex_N));
	getLastCudaError("Kernel execution failed");

	printf("%f\n", Nbeq[0]);
	printf("%f\n", Nbeq[c- 1]);
                                                                                
    //cudaFreeHost(N);                                                            
    free(N);                                                            
	checkCudaErrors(cudaFreeArray(cuArray_N));

    cudaFreeHost(beq);                                                          
    cudaFreeHost(Nbeq);                                                         
    cudaFreeHost(Nbeq_partial);  

}

//------------------------------------------------------------------------//
// version 2
//------------------------------------------------------------------------//
void run_ver2()
{
	float *N;                                                                   
    float *beq;                                                                 
    float *Nbeq;                                                                
                                                                                
    int rows = 1000;                                                            
    int cols = 1000;                                                            
                                                                                
    int r = (rows == 1) ? 1 : BLK(rows, 32) * 32;                               
    int c = (cols == 1) ? 1 : BLK(cols, 32) * 32;                               
                                                                                
    // x cols, y rows                                                           
    dim3 blkdim= dim3(32, 32, 1);                                           
    dim3 grddim= dim3(1,  BLK(r,32), 1);                            
                                                                                
    cudaMallocManaged((void**)&N,   r * c * FLT_SIZE);                          
    cudaMallocManaged((void**)&beq, c * FLT_SIZE);                              
    cudaMallocManaged((void**)&Nbeq, c * FLT_SIZE);                             


	for(int i=0; i<r; i++) {
		for(int j=0; j<c; j++) {
			N[i * c + j] = 1.f;
		}	
	}

	for(int j=0; j<c; j++) {
		beq[j] = 3.f;
	}	
                                                                                
    // gpu timer                                                                
    float gpuTime;                                                              
    cudaEvent_t startEvent, stopEvent;                                          
    checkCudaErrors( cudaEventCreate(&startEvent) );                            
    checkCudaErrors( cudaEventCreate(&stopEvent) );                             
    cudaEventRecord(startEvent);                                                

    kernel_v2 <<< grddim, blkdim >>> (N,                  
            beq,                                                                
            c,                                                                  
			c>>5,
            Nbeq);                                                      

                                                                                
    cudaEventRecord(stopEvent);                                                 
    cudaEventSynchronize(stopEvent);                                            
    cudaEventElapsedTime(&gpuTime,startEvent,stopEvent);                        
    printf("=>\t\t\tGPU execution time : %f ms\n", gpuTime);                    
                                                                                
                                                                                
    cudaDeviceSynchronize();                                                    

	printf("%f\n", Nbeq[0]);
	printf("%f\n", Nbeq[c- 1]);
                                                                                
    cudaFreeHost(N);                                                            
    cudaFreeHost(beq);                                                          
    cudaFreeHost(Nbeq);                                                         
}


//------------------------------------------------------------------------//
// version 3
//------------------------------------------------------------------------//
void run_ver3()
{
	float *N;                                                                   
    float *beq;                                                                 
    float *Nbeq_partial;                                                        
    float *Nbeq;                                                                
                                                                                
    int rows = 1000;                                                            
    int cols = 1000;                                                            
                                                                                
    int r = (rows == 1) ? 1 : BLK(rows, 32) * 32;                               
    int c = (cols == 1) ? 1 : BLK(cols, 32) * 32;                               
                                                                                
    // x cols, y rows                                                           
    dim3 Blk_part1 = dim3(16, 32, 1);                                           
    dim3 Grd_part1 = dim3(BLK(c, 16), BLK(r,32), 1);                            
                                                                                
    dim3 Blk_part2 = dim3(32, 1, 1);                                            
    dim3 Grd_part2 = dim3(BLK(r, 32), 1, 1);                                    
                                                                                
    int partial_r = r;                                                          
    int partial_c = Grd_part1.x;                                                
                                                                                
    cudaMallocManaged((void**)&N,   r * c * FLT_SIZE);                          
    cudaMallocManaged((void**)&beq, c * FLT_SIZE);                              
    cudaMallocManaged((void**)&Nbeq, c * FLT_SIZE);                             
    cudaMallocManaged((void**)&Nbeq_partial , partial_r * partial_c * FLT_SIZE);


	for(int i=0; i<r; i++) {
		for(int j=0; j<c; j++) {
			N[i * c + j] = 1.f;
		}	
	}

	for(int j=0; j<c; j++) {
		beq[j] = 3.f;
	}	
                                                                                
    // gpu timer                                                                
    float gpuTime;                                                              
    cudaEvent_t startEvent, stopEvent;                                          
    checkCudaErrors( cudaEventCreate(&startEvent) );                            
    checkCudaErrors( cudaEventCreate(&stopEvent) );                             
    cudaEventRecord(startEvent);                                                
                                                                                
	//------------------------------------------------------------------------//
	// version 1
	//------------------------------------------------------------------------//
    kernel_v3_part1 <<< Grd_part1, Blk_part1 >>> (N,                  
            beq,                                                                
            r,                                                                  
            c,                                                                  
            Nbeq_partial);                                                      
                                                                                
                                                                                
    kernel_v3_part2 <<< Grd_part2, Blk_part2 >>> (                    
            Nbeq_partial,                                                       
            partial_c,                                                          
            Nbeq);                                                              
                                                                                
                                                                                
    cudaEventRecord(stopEvent);                                                 
    cudaEventSynchronize(stopEvent);                                            
    cudaEventElapsedTime(&gpuTime,startEvent,stopEvent);                        
    printf("=>\t\t\tGPU execution time : %f ms\n", gpuTime);                    
                                                                                
                                                                                
    cudaDeviceSynchronize();                                                    

	printf("%f\n", Nbeq[0]);
	printf("%f\n", Nbeq[c- 1]);
                                                                                
    cudaFreeHost(N);                                                            
    cudaFreeHost(beq);                                                          
    cudaFreeHost(Nbeq);                                                         
    cudaFreeHost(Nbeq_partial);  
}
