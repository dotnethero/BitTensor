using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

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
        cudaMallocAsync(&pointer, bytes, CuStream.Default);
        return pointer;
    }
    
    public static void Free(void* pointer)
    {
        cudaFreeAsync(pointer, CuStream.Default);
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
            cudaMemcpyAsync(dp, Pointer, bytes, cudaMemcpyKind.cudaMemcpyDeviceToHost, CuStream.Default);
        }
    }

    public void CopyToDevice(ReadOnlySpan<T> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        var bytes = (uint)(Size * ElementSize);
        fixed (T* sp = source)
        {
            cudaMemcpyAsync(Pointer, sp, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice, CuStream.Default);
        }
    }

    public void CopyToDevice(T* source, int offset, int size)
    {
        var bytes = (uint)(Size * ElementSize);
        cudaMemcpyAsync(Pointer + offset, source, bytes, cudaMemcpyKind.cudaMemcpyHostToDevice, CuStream.Default);
    }

    public void Dispose()
    {
        cudaFreeAsync(Pointer, CuStream.Default);
    }
}
