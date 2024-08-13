#pragma once

#include <cuda_runtime.h>

extern "C"
{
    void f32_memset(int size, float* a, float value, cudaStream_t stream = nullptr);
    void f32_sigmoid_forward(int size, const float* a, float* output, cudaStream_t stream = nullptr);
    void f32_sigmoid_backward(int size, const float* g, const float* a, float* output, cudaStream_t stream = nullptr);
    void f32_tanh_forward(int size, const float* a, float* output, cudaStream_t stream = nullptr);
    void f32_tanh_backward(int size, const float* g, const float* a, float* output, cudaStream_t stream = nullptr);
    void f32_relu(int size, const float* a, float* output, cudaStream_t stream = nullptr);
    void f32_leaky_relu(int size, const float* a, float* output, float alpha, cudaStream_t stream = nullptr);
}