using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly;

public unsafe partial class CuTensor : AbstractTensor, IDisposable
{
    private readonly float* _pointer;

    public CuTensor(int[] shape) : base(shape)
    {
        _pointer = CuArray.Allocate(Size);
    }

    public CuTensor(int[] shape, float[] values) : base(shape)
    {
        _pointer = CuArray.Allocate(Size, values);
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

        CuArray.CopyToHost(_pointer, destination, Size);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        CuArray.CopyToDevice(source, _pointer, Size);
    }
    
    public void Dispose()
    {
        CuArray.Free(_pointer);
    }
}
