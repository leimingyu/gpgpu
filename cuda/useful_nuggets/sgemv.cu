inline int BLK(int number, int blksize)                                         
{                                                                               
    return (number + blksize - 1) / blksize;                                    
}

//---------------------------------------------------------------------------//
// sgemv
// 	C = A * B
//---------------------------------------------------------------------------//
#include <helpler_math.h>

__global__ void kernel_sgemv_2d (const int rows,
		const int cols,
		const int col_iters,
		const float* __restrict__ A,
		const float* __restrict__ B,
		float* __restrict__ C)
{
	int gx  = threadIdx.x;
	int gy  = threadIdx.y + __mul24(blockIdx.y, blockDim.y); // rows

	// 2x work
	gy = (gy << 1);

	int lane_id = threadIdx.x & 0x1F;

	int row_idx  = gy * cols;

	float2 tmp = make_float2(0.f, 0.f);

	float2 A_vec;
	float2 preA;
	float2 preA1;

	// each iteration, x4 work
	for(int i=0; i<col_iters; i++)
	{
		//int curr_col  = gx + i * 128;
		int curr_col  = gx + (i<<7);
		int curr_col1 = curr_col + 32;
		int curr_col2 = curr_col + 64;
		int curr_col3 = curr_col + 96;

		float b;
		float b1;
		float b2;
		float b3;

		int addr;
		int addr1;
		int addr2;
		int addr3;

		// prefetch 1
		if (curr_col1 < cols) {
			b1 = B[curr_col1];
			addr1 = row_idx + curr_col1;
			preA = make_float2(A[addr1], A[addr1 + cols]);
		}

		// work 
		if (curr_col < cols) {
			b = B[curr_col];
			addr = row_idx + curr_col;

			A_vec = make_float2(A[addr], A[addr + cols]);
			tmp += A_vec * b;
		}

		// prefetch 2
		if (curr_col2 < cols) {
			b2    = B[curr_col2];
			addr2 = row_idx + curr_col2;
			preA1 = make_float2(A[addr2], A[addr2 + cols]);
		}

		// work 1
		if (curr_col1 < cols) {
			tmp += preA * b1;
		}

		// prefetch 3
		if (curr_col3 < cols) {
			b3    = B[curr_col3];
			addr3 = row_idx + curr_col3;
			preA = make_float2(A[addr3], A[addr3 + cols]);
		}

		// work 2
		if (curr_col2 < cols) {
			tmp += preA1 * b2;
		}

		// work 3
		if (curr_col3 < cols) {
			tmp += preA * b3;
		}
	}

	// warp reduction on tmp	
	tmp.x  += __shfl_down(tmp.x,  16, 32); 
	tmp.y  += __shfl_down(tmp.y,  16, 32);

	tmp.x  += __shfl_down(tmp.x,   8, 32);                                      
	tmp.y  += __shfl_down(tmp.y,   8, 32);                                      

	tmp.x  += __shfl_down(tmp.x,   4, 32);                                      
	tmp.y  += __shfl_down(tmp.y,   4, 32);                                      

	tmp.x  += __shfl_down(tmp.x,   2, 32);                                      
	tmp.y  += __shfl_down(tmp.y,   2, 32);                                      

	tmp.x  += __shfl_down(tmp.x,   1, 32);
	tmp.y  += __shfl_down(tmp.y,   1, 32);                                      

	if(lane_id == 0) {
		C[gy]      = tmp.x;
		C[gy + 1]  = tmp.y;
	}

}


void cuda_sgemv_2d(float *d_Amat, const int A_rows, const int A_cols,
		float *d_Bvec, float d_Cvec)
{
	dim3 Blk_config = dim3(32, 4, 1);                                           
	dim3 Grd_config = dim3(1, BLK(A_rows/2, 4), 1);

	kernel_sgemv_2d <<< Grd_config, Blk_config>>>(A_rows, 
			A_cols, 
			BLK(A_cols, 128),
			d_Amat,
			d_Bvec,
			d_Cvec);
}


//---------------------------------------------------------------------------//
// sgemm
//---------------------------------------------------------------------------//
