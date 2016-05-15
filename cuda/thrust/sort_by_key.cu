#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> 

#include <stdio.h>

int main()
{
	const int N = 6;
	
	thrust::host_vector<float> h_keys(N);                                 
	thrust::host_vector<float> h_keysSorted(N);                           
	thrust::host_vector<char> h_values(N); 

	//float keys[N] = {  1.f,   4.f,   1.1f,   1.f,   5.f,   4.f};
	//char values[N] = {'a', 'b', 'a', 'a', 'e', 'b'};

	h_keys[0] = 1.f;
	h_keys[1] = 4.f;
	h_keys[2] = 1.1f;
	h_keys[3] = 1.f;
	h_keys[4] = 5.f;
	h_keys[5] = 4.f;

	h_values[0] = 'a';
	h_values[1] = 'b';
	h_values[2] = 'a';
	h_values[3] = 'a';
	h_values[4] = 'e';
	h_values[5] = 'b';

	printf("before sorting\n");
	for(int i=0; i<6; i++)
	{
		printf("key : %f \t value: %c\n", h_keys[i], h_values[i]);			
	}

	thrust::device_vector<float> d_keys;                                            
	thrust::device_vector<char> d_values; 

	d_keys = h_keys;
	d_values = h_values;

	// gpu sorting
	thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

	// copy data back to cpu
	thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());
	thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

	printf("\nafter sorting\n");
	for(int i=0; i<6; i++)
	{
		printf("key : %f \t value: %c\n", h_keys[i], h_values[i]);			
	}

	return 0;
}
