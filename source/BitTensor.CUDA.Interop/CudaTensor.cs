using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe class CudaTensor<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged
{
    public readonly CudaArray<T> Array;

    public T* Pointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Array.Pointer;
    }

    public CudaTensor(Shape shape) : base(shape)
    {
        Array = CudaArray.Allocate<T>(shape.ArraySize);
    }
    
    public CudaTensor(Shape shape, ReadOnlySpan<T> values) : base(shape)
    {
        Array = CudaArray.Allocate<T>(values);
    }

    private CudaTensor(Shape shape, CudaArray<T> array) : base(shape)
    {
        Array = array;
    }

    public CudaTensor<T> Reshape(Shape shape) // no allocation
    {
        Shape.EnsureCanReshape(shape);
        return new(shape, Array);
    }
    
    public CudaTensor<T> Transpose(Index[] axis) // no allocation
    {
        var shape = Shape.Transpose(axis);
        return new(shape, Array);
    }

    int IDeviceArray.ElementSize => Array.ElementSize;
    int IDeviceArray.Size => Array.Size;

    public void CopyToHost(Span<T> destination) => Array.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<T> source) => Array.CopyToDevice(source);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";

    public void Dispose() => Array.Dispose();
}
