using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe class CuTensor<T> : AbstractTensor, IDeviceArray<T>, IHasContext where T : unmanaged
{
    public readonly CuContext Context;
    public readonly CuArray<T> Array;

    // TODO: inline
    public T* Pointer => Array.Pointer;

    int IDeviceArray<T>.ElementSize => Array.ElementSize;
    long IDeviceArray<T>.Size => Array.Size;

    CuContext IHasContext.GetContext() => Context;

    public CuTensor(CuContext context, CuArray<T> array, Shape shape) : base(shape)
    {
        Context = context;
        Array = array;
    }

    public CuTensor<T> Reshape(Shape shape) // no allocation
    {
        if (shape.ArraySize != Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        return new(Context, Array, shape);
    }
    
    public void CopyToHost(Span<T> destination) => Array.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<T> source) => Array.CopyToDevice(source);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
