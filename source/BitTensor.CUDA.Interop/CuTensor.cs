using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe class CuTensor<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged
{
    public readonly CuArray<T> Array;

    public T* Pointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Array.Pointer;
    }

    public CuTensor(Shape shape) : base(shape)
    {
        Array = CuArray.Allocate<T>(shape.ArraySize);
    }
    
    public CuTensor(Shape shape, ReadOnlySpan<T> values) : base(shape)
    {
        Array = CuArray.Allocate<T>(values);
    }

    public CuTensor(Shape shape, CuArray<T> array) : base(shape)
    {
        Array = array;
    }

    public CuTensor<T> Reshape(Shape shape) // no allocation
    {
        if (shape.ArraySize != Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        return new(shape, Array);
    }
    
    int IDeviceArray<T>.ElementSize => Array.ElementSize;
    int IDeviceArray<T>.Size => Array.Size;

    public void CopyToHost(Span<T> destination) => Array.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<T> source) => Array.CopyToDevice(source);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
