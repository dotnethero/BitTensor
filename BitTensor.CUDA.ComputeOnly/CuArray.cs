﻿using System.Runtime.CompilerServices;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly;

using static cudaRT;

public static unsafe class CuArray
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Bytes(int size) => (uint)(size * sizeof(float));

    public static float* Allocate(int size)
    {
        void* pointer;
        cudaMalloc(&pointer, Bytes(size));
        return (float*)pointer;
    }

    public static float* Allocate(int size, float[] values)
    {
        var pointer = Allocate(size);
        CopyToDevice(values, pointer, size);
        return pointer;
    }
    
    public static void CopyToHost(float* source, Span<float> destination, int size)
    {
        fixed(float* dp = destination)
            cudaMemcpy(source, dp, Bytes(size), cudaMemcpyKind.cudaMemcpyDeviceToHost);
    }
    
    public static void CopyToDevice(ReadOnlySpan<float> source, float* destination, int size)
    {
        fixed(float* sp = source)
            cudaMemcpy(sp, destination, Bytes(size), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    public static void Free(void* pointer)
    {
        cudaFree(pointer);
    }
}
