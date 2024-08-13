#include <device_launch_parameters.h>
#include <cmath>

#include "kernels.cuh"

__global__ void kernel_memset(const unsigned int size, float* a, const float value)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		a[i] = value;
	}
}

__global__ void kernel_sigmoid_forward(const unsigned int size, const float* a, float* output)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		output[i] = 1.0f / (1.0f + expf(-a[i]));
	}
}

__global__ void kernel_sigmoid_backward(const unsigned int size, const float* grad, const float* sigmoid, float* output)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		output[i] = grad[i] * sigmoid[i] * (1.0f - sigmoid[i]);
	}
}

__global__ void kernel_tanh_forward(const unsigned int size, const float* a, float* output)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		output[i] = tanhf(a[i]);
	}
}

__global__ void kernel_tanh_backward(const unsigned int size, const float* grad, const float* tanh, float* output)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		const auto t = tanh[i];
		output[i] = grad[i] * (1.0f - t * t);
	}
}

__global__ void kernel_relu(const unsigned int size, const float* a, float* output) {
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		const auto t = a[i];
		output[i] = t > 0 ? t : 0;
	}
}

__global__ void kernel_leaky_relu(const unsigned int size, const float* a, float* output, const float alpha)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		const auto t = a[i];
		output[i] = t > 0 ? t : alpha * t;
	}
}

extern "C"
{
	constexpr int block_size = 256;

	constexpr int get_block_number(const int size)
	{
		return (size + block_size - 1) / block_size;
	}

	void f32_memset(const int size, float* a, const float value, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_memset<<<block_number, block_size, 0, stream>>>(size, a, value);
	}

	void f32_sigmoid_forward(const int size, const float* a, float* output, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_sigmoid_forward<<<block_number, block_size, 0, stream>>>(size, a, output);
	}

	void f32_sigmoid_backward(const int size, const float* g, const float* a, float* output, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_sigmoid_backward<<<block_number, block_size, 0, stream>>>(size, g, a, output);
	}

	void f32_tanh_forward(const int size, const float* a, float* output, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_tanh_forward<<<block_number, block_size, 0, stream>>>(size, a, output);
	}

	void f32_tanh_backward(const int size, const float* g, const float* a, float* output, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_tanh_backward<<<block_number, block_size, 0, stream>>>(size, g, a, output);
	}

	void f32_relu(const int size, const float* a, float* output, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_relu<<<block_number, block_size, 0, stream>>>(size, a, output);
	}

	void f32_leaky_relu(const int size, const float* a, float* output, const float alpha, cudaStream_t stream)
	{
		const int block_number = get_block_number(size);
		kernel_leaky_relu<<<block_number, block_size, 0, stream>>>(size, a, output, alpha);
	}
}
