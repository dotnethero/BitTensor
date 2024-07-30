using System.Runtime.CompilerServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Abstractions;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace BitTensor.CUDA;

using DTypeBuffer = MemoryBuffer1D<float, Stride1D.Dense>;
using DTypeView = ArrayView<float>;

public partial class CuTensor : 
    AbstractTensorNode<CuTensor>, 
    ITensorNode<CuTensor>, 
    ITensor<CuTensor>,
    IMutableTensor<CuTensor>,
    IHasAllocator<CuTensor>,
    IDeviceArray
{
    internal readonly CudaAccelerator Accelerator;
    internal readonly DTypeBuffer ArrayBuffer;
    
    internal DTypeView ArrayView
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ArrayBuffer.View;
    }
    
    internal unsafe void* Pointer
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ArrayBuffer.NativePtr.ToPointer();
    }

    public ITensorAllocator<CuTensor> Allocator { get; }

    internal CuTensor(CudaAccelerator accelerator, int[] shape) : base(shape)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        ArrayBuffer = accelerator.Allocate1D<float>(Size);
    }

    internal CuTensor(CudaAccelerator accelerator, int[] shape, float[] values) : base(shape)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        ArrayBuffer = accelerator.Allocate1D(values);
    }

    internal CuTensor(CudaAccelerator accelerator, int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward) : base(shape, children, forward, backward)
    {
        Accelerator = accelerator;
        Allocator = new CuAllocator(accelerator);
        ArrayBuffer = accelerator.Allocate1D<float>(Size);
    }

    // Reshape
    internal CuTensor(int[] shape, CuTensor tensor) : base(shape, [tensor], _ => {}, (grad, self) => [CreateReshape(tensor.Shape, grad)])
    {
        Accelerator = tensor.Accelerator;
        Allocator = tensor.Allocator;
        ArrayBuffer = tensor.ArrayBuffer;
    }

    public static CuTensor CreateNode(int[] shape, CuTensor[] children, ForwardFunction forward, BackwardFunction backward)
    {
        return new CuTensor(children[0].Accelerator, shape, children, forward, backward);
    }

    public static CuTensor CreateReshape(int[] shape, CuTensor source)
    {
        return new CuTensor(shape, source);
    }

    public void CopyToHost(Span<float> destination)
    {
        if (destination.Length != Size)
            throw new ArgumentException($"Destination array size ({destination.Length}) not equal to allocated array size ({Size})");

        EnsureHasUpdatedValues();
        ArrayView.CopyToCPU(destination);
    }

    public void CopyToDevice(ReadOnlySpan<float> source)
    {
        if (source.Length != Size)
            throw new ArgumentException($"Source array size ({source.Length}) not equal to allocated array size ({Size})");

        ArrayView.CopyFromCPU(source);
    }

    public void Dispose()
    {
        ArrayBuffer.Dispose();
    }
}
