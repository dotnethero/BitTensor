using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public abstract unsafe class AbstractNode<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged, IFloatingPoint<T>
{
    public abstract CudaContext Context { get; }
    public abstract CudaTensor<T> Tensor { get; }

    public T* Pointer => Tensor.Pointer;

    int IDeviceArray.ElementSize => sizeof(T);
    int IDeviceArray.Size => Shape.ArraySize;

    protected AbstractNode(Shape shape) : base(shape)
    {
    }

    // non-allocating transormations
    public AbstractNode<T> Reshape(Shape shape) => new Reshape<T>(this, shape);
    public AbstractNode<T> Transpose(Index[] axis) => new Permutation<T>(this, axis);
    public AbstractNode<T> Transpose() => Transpose(Shape.GetTransposeAxis());

    public void CopyToHost(Span<T> destination) => Tensor.CopyToHost(destination);
    public void CopyToDevice(ReadOnlySpan<T> source) => Tensor.CopyToDevice(source);

    public virtual void DisposeResources() {}

    public void Dispose()
    {
        DisposeResources();
    }
}
