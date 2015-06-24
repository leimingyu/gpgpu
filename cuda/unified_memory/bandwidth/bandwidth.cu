#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#define MEMCOPY_ITERATIONS  10                                                  
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M                      
#define DEFAULT_INCREMENT   (1 << 22)               //4 M                       
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M     

#define MEMSIZE_MAX     (1 << 26)         //64 M                          
#define MEMSIZE_START   (1 << 10)         //1 KB                          
#define INCREMENT_1KB   (1 << 10)         //1 KB                          
#define INCREMENT_2KB   (1 << 11)         //2 KB                          
#define INCREMENT_10KB  (10 * (1 << 10))  //10KB                          
#define INCREMENT_100KB (100 * (1 << 10)) //100 KB                        
#define INCREMENT_1MB   (1 << 20)         //1 MB                          
#define INCREMENT_2MB   (1 << 21)         //2 MB                          
#define INCREMENT_4MB   (1 << 22)         //4 MB                          
#define LIMIT_20KB      (20 * (1 << 10))  //20 KB                         
#define LIMIT_50KB      (50 * (1 << 10))  //50 KB                         
#define LIMIT_100KB     (100 * (1 << 10)) //100 KB                        
#define LIMIT_1MB       (1 << 20)         //1 MB                          
#define LIMIT_16MB      (1 << 24)         //16 MB                         
#define LIMIT_32MB      (1 << 25)         //32 MB    


enum MemMode {PINNED, PAGEABLE, UM};
enum CopyType {HOST2DEVICE, DEVICE2HOST, DEVICE2DEVICE};

const char *MemoryCopyStr[] =                                                 
{                                                                               
    "Device to Host",                                                           
    "Host to Device",                                                           
    "Device to Device",                                                         
    NULL                                                                        
};                                                                              
                                                                                
const char *MemoryModeStr[] =                                                     
{                                                                               
    "PINNED",                                                                   
    "PAGEABLE",                                                                 
    NULL                                                                        
};    


float elapsedTimeInMs = 0.0f;                                               
float bandwidthInMBs = 0.0f;                                                

cudaError_t err;
cudaEvent_t start, stop;   


/*
struct DataElement
{
	char *name;
	int value;
};

__global__ 
void Kernel(DataElement *elem) {
	printf("On device: name=%s, value=%d\n", elem->name, elem->value);

	elem->name[0] = 'd';
	elem->value++;
}

void launch(DataElement *elem) {
	Kernel<<< 1, 1 >>>(elem);
	cudaDeviceSynchronize();
}
*/
void StartDevice(int device)
{
	cudaDeviceProp DevProp;
	err = cudaGetDeviceProperties(&DevProp, device);
	if (err == cudaSuccess)                                            
	{                                                                       
		printf(" Device %d: %s\n", device, DevProp.name);         

		if (DevProp.computeMode == cudaComputeModeProhibited)            
		{                                                                   
			fprintf(stderr, "Error: current status is <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
			checkCudaErrors(cudaSetDevice(device));                  
			cudaDeviceReset();                                              
			exit(EXIT_FAILURE);                                             
		}                                                                   
	}                                                                       
	else                                                                    
	{                                                                       
		printf("Failed! Return with error : %d\n-> %s\n", (int)err, cudaGetErrorString(err));
		checkCudaErrors(cudaSetDevice(device));                      
		cudaDeviceReset();                                                  
		exit(EXIT_FAILURE);                                                 
	}                                    
}

float Run_H2D(unsigned int transfer_size, MemMode mem_mode)
{
	elapsedTimeInMs = 0.0f;                                               
	bandwidthInMBs = 0.0f;                                                

	//allocate host memory                                                      
	unsigned char *h_odata = NULL;                                              

	if (PINNED == mem_mode)                                                      
	{
		checkCudaErrors(cudaHostAlloc((void **)&h_odata, transfer_size, cudaHostAllocWriteCombined));
	}
	else                                                                        
	{
		h_odata = (unsigned char *)malloc(transfer_size);                             
		if (h_odata == 0)                                                       
		{                                                                       
			fprintf(stderr, "Error: not enough host memory available!\n");
			exit(EXIT_FAILURE);                                                 
		}                                                                       

	}

	// 16M 
	unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);   
	unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);   

	if (h_cacheClear1 == 0 || h_cacheClear1 == 0)   
	{
		fprintf(stderr, "Error: not enough host memory available!\n");
		exit(EXIT_FAILURE);                                                 
	}

	//initialize the memory                                                     
	for (unsigned int i = 0; i < transfer_size/sizeof(unsigned char); i++)            
	{                                                                           
		h_odata[i] = (unsigned char)(i & 0xff);                                 
	}                                                                           

	for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) 
	{                                                                           
		h_cacheClear1[i] = (unsigned char)(i & 0xff);                           
		h_cacheClear2[i] = (unsigned char)(0xff - (i & 0xff));                  
	}                                                         

	//allocate device memory                                                    
	unsigned char *d_idata;                                                     
	checkCudaErrors(cudaMalloc((void **) &d_idata, transfer_size));                   

	checkCudaErrors(cudaEventRecord(start, 0));  

	//copy host memory to device memory                                         
	if (PINNED == mem_mode)                                                      
	{                                                                           
		for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)                   
			checkCudaErrors(cudaMemcpyAsync(d_idata, h_odata, transfer_size, cudaMemcpyHostToDevice, 0));        
	}                                                                           
	else                                                                        
	{                                                                           
		for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)                   
			checkCudaErrors(cudaMemcpy(d_idata, h_odata, transfer_size, cudaMemcpyHostToDevice));                
	}                                 

	checkCudaErrors(cudaEventRecord(stop, 0));                                  
	checkCudaErrors(cudaDeviceSynchronize());                                   
	//total elapsed time in ms                                                  
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));     


	//calculate bandwidth in MB/s                                               
	bandwidthInMBs = ((float)(1<<10) * transfer_size * (float)MEMCOPY_ITERATIONS) /
		(elapsedTimeInMs * (float)(1 << 20));                      



	if (PINNED == mem_mode)                                                      
		checkCudaErrors(cudaFreeHost(h_odata));                                 
	else                                                                        
		free(h_odata);                                                          

	free(h_cacheClear1);                                                        
	free(h_cacheClear2);                                                        
    	checkCudaErrors(cudaFree(d_idata));

	return bandwidthInMBs;    
}
		/*
		   DataElement *e;

		   e->value = 10;
		   cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
		   strcpy(e->name, "hello");

		   launch(e);

		   printf("On host: name=%s, value=%d\n", e->name, e->value);

		   cudaFree(e->name);
		   cudaFree(e);
		 */

float Run_D2H(unsigned int transfer_size, MemMode mem_mode)
{
	elapsedTimeInMs = 0.0f;                                               
	bandwidthInMBs = 0.0f;                                                
/*
unsigned char *h_idata = NULL;                                              
unsigned char *h_odata = NULL; 

	if(mem_mode == PINNED)
	{
		checkCudaErrors(cudaHostAlloc((void **)&h_idata, transfer_size, cudaHostAllocWriteCombined));
		checkCudaErrors(cudaHostAlloc((void **)&h_odata, transfer_size, cudaHostAllocWriteCombined));
	}
	else if (mem_mode == PAGEABLE)
	{
		h_idata = (unsigned char *)malloc(transfer_size);                             
		h_odata = (unsigned char *)malloc(transfer_size);    	
		if (h_idata == 0 || h_odata == 0)
		{
			fprintf(stderr, "Error : Not enough host memory avaialable\n");
			exit(EXIT_FAILURE);      
		}
	}
	else  // unified memory
	{
		cudaMallocManaged((void**)&h_idata, transfer_size);
		cudaMallocManaged((void**)&h_odata, transfer_size);
	}

	// initialize host memory
	for (unsigned int i = 0; i < transfer_size/sizeof(unsigned char); i++)            
	{                                                                           
		h_idata[i] = (unsigned char)(i & 0xff);                                 
	}  


	// allocate device memory                                                   
	unsigned char *d_idata;                                                     
	checkCudaErrors(cudaMalloc((void **) &d_idata, transfer_size));                   

	//initialize the device memory                                              
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, transfer_size,                       
				cudaMemcpyHostToDevice));


*/
	return bandwidthInMBs;    
}

float Run_D2D(unsigned int transfer_size, MemMode mem_mode)
{
	elapsedTimeInMs = 0.0f;                                               
	bandwidthInMBs = 0.0f;                                                

	return bandwidthInMBs;    
}

void Display(unsigned int *records, double *bw, unsigned int count, CopyType copy_type, MemMode mem_mode)
{
	printf(" %s Bandwidth\n", MemoryCopyStr[copy_type]);   
	printf(" %s Memory Transfers\n", MemoryModeStr[mem_mode]);  

	printf(" Write-Combined Memory Writes are Enabled.\n");

	printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");                      
	unsigned int i;                                                             

	for (i = 0; i < (count - 1); i++)                                           
	{                                                                           
		printf("   %u\t\t\t%s%.1f\n", records[i], (records[i] < 10000)? "\t" : "", bw[i]);
	}                                                                           

	printf("   %u\t\t\t%s%.1f\n\n", records[i], (records[i] < 10000)? "\t" : "", bw[i]);
}

void RunTest(MemMode  mem_mode, CopyType copy_type)
{
	// calculate the number of records
	unsigned int count = 1 + (LIMIT_20KB  / INCREMENT_1KB)          
		+ ((LIMIT_50KB - LIMIT_20KB) / INCREMENT_2KB)
		+ ((LIMIT_100KB - LIMIT_50KB) / INCREMENT_10KB)
		+ ((LIMIT_1MB - LIMIT_100KB) / INCREMENT_100KB)
		+ ((LIMIT_16MB - LIMIT_1MB) / INCREMENT_1MB)
		+ ((LIMIT_32MB - LIMIT_16MB) / INCREMENT_2MB)
		+ ((MEMSIZE_MAX - LIMIT_32MB) / INCREMENT_4MB);

	// allocate the transfer size and bandwidth array
	unsigned int *records = (unsigned int *) malloc(count * sizeof(unsigned int));	
	double *bw = (double *) malloc(count * sizeof(double));	

	int iteration = 0;
	unsigned int transfer_size = 0;	

	while (transfer_size <= MEMSIZE_MAX)
	{
		// increase the transfering size step by step
		if (transfer_size < LIMIT_20KB)                                     
			transfer_size += INCREMENT_1KB;                                 

		else if (transfer_size < LIMIT_50KB)                                
			transfer_size += INCREMENT_2KB;                                 

		else if (transfer_size < LIMIT_100KB)                               
			transfer_size += INCREMENT_10KB;                                

		else if (transfer_size < LIMIT_1MB)                                 
			transfer_size += INCREMENT_100KB;                               

		else if (transfer_size < LIMIT_16MB)                                
			transfer_size += INCREMENT_1MB;                                 

		else if (transfer_size < LIMIT_32MB)                                
			transfer_size += INCREMENT_2MB;                                 

		else                                                          
			transfer_size += INCREMENT_4MB;                                 

		records[iteration] = transfer_size;	

		// run test according to the copy type
		switch (copy_type)
		{ 
			case HOST2DEVICE:
				bw[iteration] += Run_H2D(transfer_size, mem_mode);
				break;

			case DEVICE2HOST:
				bw[iteration] += Run_D2H(transfer_size, mem_mode);
				break;

			case DEVICE2DEVICE:
				bw[iteration] += Run_D2D(transfer_size, mem_mode);
				break;
		}
		
		// next round
		iteration++;
		printf("/");
	}

	printf("\n");


	// print resutls to std out
	Display(records, bw, count, copy_type, mem_mode);

	free(records);
	free(bw);
}

int main(void)
{
	// default setup
	//MemMode mem_mode = PINNED;	
	MemMode mem_mode = PAGEABLE;	

	bool h2d = true;                                                          
	bool d2h = false;                                                          
	bool d2d = false;   

	// use 1st device
	int device = 0;
	StartDevice(device);

#if CUDART_VERSION >= 2020                                                      
	printf("Current memory mode %d\n", mem_mode);                                                                                
#endif   


	// initialization
	checkCudaErrors(cudaEventCreate(&start));                                   
	checkCudaErrors(cudaEventCreate(&stop));

	// run tests
	if (h2d)
		RunTest(mem_mode, HOST2DEVICE);

	if (d2h)
		RunTest(mem_mode, DEVICE2HOST);

	if (d2d)
		RunTest(mem_mode, DEVICE2DEVICE);
	
	// clean up
	checkCudaErrors(cudaEventDestroy(stop));                                    
	checkCudaErrors(cudaEventDestroy(start)); 

	cudaDeviceReset();
	return 0;
}
