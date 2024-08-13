
#include <cuda_runtime.h>

#include "helpers.hpp"
#include "kernels.cuh"

int main()
{
	constexpr int size = 5;

	constexpr float a[size] = { -1, -2, 3, 4, 5 };
    constexpr float b[size] = {  1, -1, 1, 2, 0 };

    const auto da = f32_allocate(size, a);
    const auto db = f32_allocate(size, b);
    const auto dc = f32_allocate(size);
    const auto dd = f32_allocate(size);

	ensure_success(cudaSetDevice(0));

    // f32_memset(size, dc, 5);
    // f32_leaky_relu(size, da, dc, 0.01f);
    // f32_relu(size, da, dc);

    f32_tanh_forward(size, da, dc);
    f32_tanh_backward(size, db, dc, dd);
    f32_print_dev(size, dc);
    f32_print_dev(size, dd);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dd);
    return 0;
}
