﻿using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using static cudaRT;

public static unsafe class CudaArray
{
    public static CudaArray<T> Allocate<T>(int size) where T : unmanaged
    {
        var bytes = (uint)(size * sizeof(T));
        var pointer = (T*)AllocateRaw(bytes);
        var array = new CudaArray<T>(pointer, size);
        return array;
    }
    
    public static CudaArray<T> Allocate<T>(ReadOnlySpan<T> values) where T : unmanaged
    {
        var array = Allocate<T>(values.Length);
        array.CopyToDevice(values);
        return array;
    }

    public static void* AllocateRaw(uint bytes)
    {
        void* pointer;
        cudaMalloc(&pointer, bytes);
        return pointer;
    }
    
    public static void Free(void* pointer)
    {
        cudaFree(pointer);
    }
}

public unsafe class CudaArray<T> : IDeviceArray<T>, IDisposable where T : unmanaged
{
    public int Size { get; }
    public int ElementSize { get; }
    public T* Pointer { get; }

    internal CudaArray(T* pointer, int size)
    {
        ElementSize = sizeof(T);
        Size = size;
        Pointer = pointer;
    }

    public void CopyToHost(Span<T> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        var bytes = (uint)(Size * ElementSize);
        fixed (T* dp = destination)
        {
            cudaMemcpy(dp, Pointer, bytes, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
    }

    public void CopyToDevice(ReadOnlySpan<T> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        var bytes = (uint)(Size * ElementSize);
        fixed (T* sp = source)
        {
            cudaMemcpy(Pointer, sp, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
    }

    public void CopyToDevice(ReadOnlySpan<T> source, int offset, int size)
    {
        if (offset < 0 || offset + size > Size)
            throw new InvalidOperationException($"Out of boundaries: offset={offset}, size={size}, array.Size={Size}");

        var bytes = (uint)(size * ElementSize);
        fixed (T* sp = source)
        {
            cudaMemcpy(Pointer + offset, sp, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
    }

    public void Dispose()
    {
        cudaFree(Pointer);
    }
}
