using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe partial class CuTensor : AbstractTensor, ITensor<CuTensor>, IDeviceArray, IDisposable
{
    internal readonly float* Pointer;

    public CuTensor(Shape shape) : base(shape)
    {
        Pointer = CuArray.Allocate(Size);
    }

    public CuTensor(Shape shape, float[] values) : base(shape)
    {
        Pointer = CuArray.Allocate(Size, values);
    }
    
    public CuTensor(Shape shape, float* pointer) : base(shape)
    {
        Pointer = pointer;
    }

    public CuTensor PadLeft() => Reshape([1, ..Shape]);

    public CuTensor PadRight() => Reshape([..Shape, 1]);

    public CuTensor Reshape(Shape shape) // no allocation
    {
        if (shape.ArraySize != Size)
            throw new InvalidOperationException($"Can't reshape {Shape} into {shape}");

        return new CuTensor(shape, Pointer);
    }

    public float[] CopyToHost()
    {
        var destination = new float[Size];
        CopyToHost(destination);
        return destination;
    }
    
    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        CuArray.CopyToHost(Pointer, destination, Size);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        CuArray.CopyToDevice(source, Pointer, Size);
    }
    
    public void Dispose()
    {
        CuArray.Free(Pointer);
    }

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
