using System.Runtime.CompilerServices;
using BitTensor.Core;
using BitTensor.Native;

namespace BitTensor.Playground;

internal readonly unsafe struct DebugDeviceAllocation : IAllocation, IDisposable
{
    private readonly nuint _size;
    private readonly float* _data;
    private readonly float[] _copy;

    public Span<float> Data
    {
        get
        {
            fixed (float* cp = _copy)
            {
                CUDA.cudaMemcpy(cp, _data, _size * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }

            return _copy;
        }
    }

    public DebugDeviceAllocation(nuint size)
    {
        float* handle;

        CUDA.cudaMalloc((void**)&handle, size * sizeof(float));

        _size = size;
        _data = handle;
        _copy = new float[_size];

        Array.Fill(_copy, -1);
    }

    public void Dispose()
    {
        CUDA.cudaFree(_data);
    }
}

internal class DebugDeviceAllocator : IAllocator
{
    private static readonly Lazy<DebugDeviceAllocator> Lazy = new(() => new DebugDeviceAllocator());

    public static DebugDeviceAllocator Instance => Lazy.Value;

    public IAllocation Allocate(int size) => new DebugDeviceAllocation((nuint)(uint)size);
}