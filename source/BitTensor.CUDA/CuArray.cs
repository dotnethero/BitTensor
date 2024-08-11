using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

using static cudaRT;

public static unsafe class CuArray
{
    public static CuArray<T> Allocate<T>(this Accelerator accelerator, int size) where T : unmanaged
    {
        var buffer = accelerator.Allocate1D<T>(size);
        return new CuArray<T>(buffer);
    }
    
    public static CuArray<T> Allocate<T>(this Accelerator accelerator, T[] values) where T : unmanaged
    {
        var buffer = accelerator.Allocate1D(values);
        return new CuArray<T>(buffer);
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

public unsafe class CuArray<T> : IDeviceArray<T> where T : unmanaged
{
    public MemoryBuffer1D<T, Stride1D.Dense> Buffer { get; }
    public long Size { get; }
    public int ElementSize { get; }
    public T* Pointer { get; }

    public CuArray(MemoryBuffer1D<T, Stride1D.Dense> buffer)
    {
        Buffer = buffer;
        Size = buffer.Length;
        ElementSize = buffer.ElementSize;
        Pointer = (T*)buffer.NativePtr;
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

    public void Dispose()
    {
        cudaFree(Pointer);
    }
}
