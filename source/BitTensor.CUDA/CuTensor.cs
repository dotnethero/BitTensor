using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe class CuTensor : AbstractTensor, IDeviceArray
{
    internal readonly CuContext Context;
    internal readonly CuArray<float> Array;

    internal float* Pointer => Array.Pointer;

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

    public float[] CopyToHost()
    {
        var destination = new float[Size];
        CopyToHost(destination);
        return destination;
    }
    
    public void CopyToHost(Span<float> destination) => Array.CopyToHost(destination);

    public void CopyToDevice(ReadOnlySpan<float> source) => Array.CopyToDevice(source);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
