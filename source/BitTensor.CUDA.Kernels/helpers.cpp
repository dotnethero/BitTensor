
#include <cuda_runtime.h>
#include <driver_types.h>
#include <iostream>

#include "helpers.hpp"

constexpr void verify(const cudaError_t status)
{
	if (status != cudaSuccess)
	{
		std::cout << "CUDA error: " << status << '\n';
	}
}

void ensure_success(const cudaError_t status)
{
	verify(status);
}

float* f32_allocate(const unsigned int size)
{
    float *devptr = nullptr;
    const auto status = cudaMalloc((void**)&devptr, size * sizeof(float));
    verify(status);
    return devptr;
}

float* f32_allocate(const unsigned int size, const float* source)
{
    const auto devptr = f32_allocate(size);
    f32_copy_to_device(size, source, devptr);
    return devptr;
}

void f32_copy_to_device(const unsigned int size, const float* source, float* destination)
{
    const auto status = cudaMemcpy(destination, source, size * sizeof(float), cudaMemcpyHostToDevice);
    verify(status);
}

void f32_copy_to_host(const unsigned int size, const float* source, float* destination)
{
    const auto status = cudaMemcpy(destination, source, size * sizeof(float), cudaMemcpyDeviceToHost);
    verify(status);
}

void f32_print(const unsigned int size, const float* array)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		std::cout << array[i];
        if (i < size - 1)
        {
	        std::cout << " ";
        }
	}
    std::cout << "\n";
}

void f32_print_dev(const unsigned int size, const float* dev_array)
{
	const auto host_array = new float[size];

    cudaDeviceSynchronize();
	
    f32_copy_to_host(size, dev_array, host_array);
    f32_print(size, host_array);

    delete[] host_array;
}
