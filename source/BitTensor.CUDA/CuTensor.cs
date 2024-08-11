using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe class CuTensor : AbstractTensor, IDeviceArray<float>
{
    public readonly CuContext Context;
    public readonly CuArray<float> Array;

    public float* Pointer => Array.Pointer;

    int IDeviceArray<float>.ElementSize => Array.ElementSize;
    int IDeviceArray<float>.Size => Array.Size;

    public CuTensor(CuContext context, CuArray<float> array, Shape shape) : base(shape)
    {
        Context = context;
        Array = array;
    }

    public CuTensor Reshape(Shape shape) // no allocation
    {
        if (shape.ArraySize != Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        return new CuTensor(Context, Array, shape);
    }
    
    public void CopyToHost(Span<float> destination) => Array.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<float> source) => Array.CopyToDevice(source);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
