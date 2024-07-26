﻿using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using static BitTensor.CUDA.Interop.cudaRT;

namespace BitTensor.Playground;

internal readonly unsafe struct DebugDeviceAllocation : IAllocation, IDisposable
{
    private readonly nuint _size;
    private readonly float* _data;
    private readonly float[] _copy;

    public float* Pointer => _data;

    public Span<float> Data
    {
        get
        {
            CopyToHost(_copy);
            return _copy;
        }
    }

    public DebugDeviceAllocation(nuint size)
    {
        float* handle;

        cudaMalloc((void**)&handle, size * sizeof(float));

        _size = size;
        _data = handle;
        _copy = new float[_size];

        Array.Fill(_copy, -1);
    }
    
    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != (int)_size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({_size})");

        fixed (float* dp = destination)
        {
            cudaMemcpy(dp, _data, _size * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
        }
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != (int)_size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({_size})");

        fixed (float* sp = source)
        {
            cudaMemcpy(_data, sp, _size * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
        }
    }

    public void Dispose()
    {
        cudaFree(_data);
    }
}

internal class DebugDeviceAllocator : IAllocator
{
    private static readonly Lazy<DebugDeviceAllocator> Lazy = new(() => new DebugDeviceAllocator());

    public static DebugDeviceAllocator Instance => Lazy.Value;

    public IAllocation Allocate(int size) => new DebugDeviceAllocation((nuint)(uint)size);
}