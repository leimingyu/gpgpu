#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>

// base class
class Managed
{
public:
	void* operator new(size_t len)
	{
		void *ptr;	
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

	void operator delete(void *ptr)
	{
		cudaFree(ptr);	
	}
};

// string class for managed 
class String : public Managed
{
public:
	String() : length(0), data(0) {}
	
	String(const char *s) : length(0) , data(0)
	{
		_realloc(strlen(s));		
		strcpy(data, s);
	}

	String(const String &s) : length(0) , data(0)
	{
		_realloc(s.length);
		strcpy(data, s.data);		
	}

	~String()
	{
		cudaFree(data);	
	}

	// assignment
	String& operator=(const char *s)
	{
		_realloc(strlen(s));		
		strcpy(data, s);
		return *this;
	}

	// access
	__host__ __device__
	char& operator[](int pos)
	{
		return data[pos];		
	}

	// string access
	__host__ __device__
	const char* c_str() const
	{
		return data;		
	}

private:
	int length;	
	char *data;

	void _realloc(int len)
	{
		cudaFree(data);		
		length = len;
		cudaMallocManaged(&data, length + 1);
	}
};



struct DataElement : public Managed
{
	//char *name;
	String name;
	int value;
};

__global__ 
void Kernel_by_pointer(DataElement *elem) 
{
	printf("On device: name=%s, value=%d\n", elem->name.c_str(), elem->value);

	elem->name[0] = 'p';
	elem->value++;
}

__global__ 
void Kernel_by_ref(DataElement &elem) 
{
	printf("On device: name=%s, value=%d\n", elem.name.c_str(), elem.value);

	elem.name[0] = 'r';
	elem.value++;
}

__global__ 
void Kernel_by_value(DataElement elem) 
{
	printf("On device: name=%s, value=%d\n", elem.name.c_str(), elem.value);

	elem.name[0] = 'k';
	elem.value++;
}

void launch_by_pointer(DataElement *elem)
{
	Kernel_by_pointer <<< 1, 1 >>> (elem);	
	cudaDeviceSynchronize();
}

void launch_by_ref(DataElement &elem)
{
	Kernel_by_ref <<< 1, 1 >>> (elem);	
	cudaDeviceSynchronize();
}


void launch_by_value(DataElement elem)
{
	Kernel_by_value <<< 1, 1 >>> (elem);	
	cudaDeviceSynchronize();
}

int main(void)
{
	DataElement *e = new DataElement; 
	e->name = "hello";
	e->value = 10;

	launch_by_pointer(e);
	printf("On host (after by-pointer): name=%s, value=%d\n", e->name.c_str(), e->value); 

	launch_by_ref(*e);
	printf("On host (after by-ref):     name=%s, value=%d\n", e->name.c_str(), e->value);

	// the change happens locally inside the kernel
	launch_by_value(*e);
	printf("On host (after by-value):     name=%s, value=%d\n", e->name.c_str(), e->value);


	cudaDeviceReset();
}
