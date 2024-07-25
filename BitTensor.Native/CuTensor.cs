using BitTensor.Abstractions;
using BitTensor.CUDA.Abstractions;

namespace BitTensor.CUDA;

public unsafe partial class CuTensor : AbstractTensorNode<CuTensor>, ITensor<CuTensor>, IDeviceArray
{
    internal readonly float* Handle;

    internal CuTensor(int[] shape) : base(shape)
    {
        Handle = CuAllocator.Allocate(Size);
    }
    
    internal CuTensor(int[] shape, float[] values) : this(shape)
    {
        CuMemory.CopyToDevice(values, Handle, Size);
    }

    internal CuTensor(int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward) : base(shape, children, forward, backward)
    {
        Handle = CuAllocator.Allocate(Size);
    }

    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        CuMemory.CopyToHost(Handle, destination, Size);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        CuMemory.CopyToDevice(source, Handle, Size);
    }

    public void Dispose()
    {
        CuAllocator.Free(Handle);
    }

}

public partial class CuTensor
{
    public static CuTensor Zero => FromValue(0);
    public static CuTensor One => FromValue(1);

    public static CuTensor FromArray(int[] shape, float[] values) => 
        new(shape, values);

    public static CuTensor FromValue(float value) =>
        FromArray(shape: [], [value]);

}