//  compute block number
inline int BLK(int number, int blksize)                                         
{                                                                               
    return (number + blksize - 1) / blksize;                                    
}                                                                               



void cudaSetVector(float *d_vec, float *h_vec, const int len)
{
	cudaError_t status;
	status = cudaMemcpy(d_vec, h_vec, sizeof(float) * len, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "cudaSetVector failed! File : %s, Line : %d.\n", 
				__FILE__, __LINE__);
		exit(1);
	}
}

void cudaSetMatrix(float *d_mat, float *h_mat, const int rows, const int cols)
{
	cudaError_t status;
	status = cudaMemcpy(d_mat, h_mat, sizeof(float) * rows * cols, cudaMemcpyHostToDevice);
	if(status != cudaSuccess) {
		fprintf(stderr, "cudaSetMatrix failed! File : %s, Line : %d.\n", 
				__FILE__, __LINE__);
		exit(1);
	}
}


//----------------------------------------------------------------------------//
// 1d 
//----------------------------------------------------------------------------//
void init1D(float *array, int rows, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		array[i] = value;                                        
	}                                                                           
}

void init1D_inc(float *array, int rows, float value)
{
	float v = value;
	for(int i=0; i<rows; i++) {
		array[i] = v + i;
	}
}

void print1D(float *data, int len)
{                                                                               
	printf("\n");
	for(int i=0; i<len; i++) {                                                 
		printf("%5.3f ", data[i]);
	}                                                                           
	printf("\n");
}

//----------------------------------------------------------------------------//
// 2d
//----------------------------------------------------------------------------//
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void init2D(float *array, int rows, int cols, float value)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			array[i * cols + j] = value;                                        
		}                                                                       
	}                                                                           
}

void print2D(float *array, int rows, int cols)
{                                                                               
	for(int i=0; i<rows; i++) {                                                 
		for(int j=0; j<cols; j++) {                                             
			printf("%8.4f ", array[i * cols + j]);
		}                                                                       
		printf("\n");
	}                                                                           
}

void init2D_colmajor(float *array, int rows, int cols, float value)
{                                                                               
	// transpose
	// store the data in column major
	float v = value;
	for (int j = 0; j < cols; j++) { 
		for (int i = 0; i < rows; i++) {   // rows of array (column major)
			array[IDX2C(i,j,rows)] = v++; 
		} 
	}
}

void print2D_colmajor(float *array, int rows, int cols, float value)
{                                                                               
	for (int i = 0; i < rows; i++) { 
		for (int j = 0; j < cols; j++) {   // rows of array (column major)
			printf("%4.f ", array[IDX2C(i,j,rows)]);
		} 
		printf("\n");
	}
}

void d2h_print1d(float *d_data, float *h_data, const int rows)
{
	cudaMemcpy(h_data, d_data, sizeof(float) * rows, cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
		printf("%f ", h_data[i]);
	}
	printf("\n");
}

void d2h_print2d(float *d_data, float *h_data, const int rows, const int cols)
{
	cudaMemcpy(h_data, d_data, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			printf("%8.4f ", h_data[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

//----------------------------------------------------------------------------//
// cublas 
//----------------------------------------------------------------------------//
void printCublas_1d(float *dev_ptr, float *hst_ptr, int len)
{
	cublasStatus_t stat;
	stat = cublasGetVector(len, sizeof(float), dev_ptr, 1, hst_ptr, 1);
	if (stat != CUBLAS_STATUS_SUCCESS) { 
		fprintf(stderr, "Failed to get data (vector) from device.\n"); 
		exit(1);
	}

	for(int i=0; i<len; i++){
		printf("%5.0f ", hst_ptr[i]);
	}	printf("\n");
}
