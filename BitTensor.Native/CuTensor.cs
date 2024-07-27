using BitTensor.Abstractions;
using BitTensor.CUDA.Abstractions;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

public unsafe partial class CuTensor : 
    AbstractTensorNode<CuTensor>, 
    ITensorNode<CuTensor>, 
    ITensor<CuTensor>, 
    IDeviceArray
{
    internal readonly Accelerator Accelerator;
    internal readonly MemoryBuffer1D<float, Stride1D.Dense> Buffer;

    internal CuTensor(Accelerator accelerator, int[] shape) : base(shape)
    {
        Accelerator = accelerator;
        Buffer = accelerator.Allocate1D<float>(Size);
    }

    internal CuTensor(Accelerator accelerator, int[] shape, float[] values) : this(accelerator, shape)
    {
        Accelerator = accelerator;
        Buffer = accelerator.Allocate1D(values);
    }

    internal CuTensor(Accelerator accelerator, int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward) : base(shape, children, forward, backward)
    {
        Accelerator = accelerator;
        Buffer = accelerator.Allocate1D<float>(Size);
    }

    public static CuTensor Create(int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward)
    {
        return new CuTensor(children[0].Accelerator, shape, children, forward, backward);
    }

    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        EnsureHasUpdatedValues();

        Buffer.View.BaseView.CopyToCPU(destination);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        Buffer.View.BaseView.CopyFromCPU(source);
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}