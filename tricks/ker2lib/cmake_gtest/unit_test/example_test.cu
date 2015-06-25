#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include "kernel.h"

TEST(ExampleTest, comparetest) 
{
	myk<<<1,1>>>();
	printf("CUDA status: %d\n", cudaDeviceSynchronize());

	int expect = 1; 
    EXPECT_EQ(1, expect);
}

