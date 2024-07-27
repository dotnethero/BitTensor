﻿using System.Runtime.CompilerServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Abstractions;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA;

public partial class CuTensor : 
    AbstractTensorNode<CuTensor>, 
    ITensorNode<CuTensor>, 
    ITensor<CuTensor>,
    IAccumulator<CuTensor>,
    IHasAllocator<CuTensor>,
    IDeviceArray
{
    internal readonly Accelerator Accelerator;
    internal readonly MemoryBuffer1D<float, Stride1D.Dense> Buffer;
    
    internal ArrayView1D<float, Stride1D.Dense> View
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Buffer.View;
    }

    public ITensorAllocator<CuTensor> Allocator { get; }

    internal CuTensor(Accelerator accelerator, int[] shape) : base(shape)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        Buffer = accelerator.Allocate1D<float>(Size);
    }

    internal CuTensor(Accelerator accelerator, int[] shape, float[] values) : this(accelerator, shape)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        Buffer = accelerator.Allocate1D(values);
    }

    internal CuTensor(Accelerator accelerator, int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward) : base(shape, children, forward, backward)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        Buffer = accelerator.Allocate1D<float>(Size);
    }

    internal CuTensor(CuTensor tensor, int[] shape) : base(shape, [tensor], self => {}, (grad, self) => [CreateReshape(grad, tensor.Shape)])
    {
        Accelerator = tensor.Accelerator;
        Allocator = tensor.Allocator;
        Buffer = tensor.Buffer;
    }

    public static CuTensor CreateNode(int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward)
    {
        return new CuTensor(children[0].Accelerator, shape, children, forward, backward);
    }

    public static CuTensor CreateReshape(CuTensor tensor, int[] shape)
    {
        return new CuTensor(tensor, shape);
    }

    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        EnsureHasUpdatedValues();
        View.BaseView.CopyToCPU(destination);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        View.BaseView.CopyFromCPU(source);
    }
    
    public void Accumulate(CuTensor value)
    {
        CuBackend.ExecuteAdd(this, value, this);
    }

    public void Dispose()
    {
        Buffer.Dispose();
    }
}