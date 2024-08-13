#pragma once

#include <driver_types.h>

void ensure_success(cudaError_t status);
float* f32_allocate(unsigned int size);
float* f32_allocate(unsigned int size, const float* source);
void f32_copy_to_device(unsigned int size, const float* source, float* destination);
void f32_copy_to_host(unsigned int size, const float* source, float* destination);
void f32_print(unsigned int size, const float* array);
void f32_print_dev(unsigned int size, const float* dev_array);
