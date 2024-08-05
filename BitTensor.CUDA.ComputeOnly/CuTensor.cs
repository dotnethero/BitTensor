using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly;

public unsafe partial class CuTensor : AbstractTensor, IDisposable
{
    internal readonly float* Pointer;

    public CuTensor(int[] shape) : base(shape)
    {
        Pointer = CuArray.Allocate(Size);
    }

    public CuTensor(int[] shape, float[] values) : base(shape)
    {
        Pointer = CuArray.Allocate(Size, values);
    }
    
    public CuTensor(int[] shape, float* pointer) : base(shape)
    {
        Pointer = pointer;
    }

    public static CuTensor Allocate(int[] shape) => new(shape);

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

    public override string ToString() => $"Tensor #{Id}, shape={Shape.Serialize()}";
}
