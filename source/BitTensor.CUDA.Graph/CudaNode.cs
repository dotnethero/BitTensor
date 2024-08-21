using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph.Nodes;

namespace BitTensor.CUDA.Graph;

public abstract unsafe partial class CudaNode<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly List<CudaNode<T>> Dependents = [];
    internal bool Outdated;

    public abstract CudaContext Context { get; }
    public abstract CudaTensor<T> Tensor { get; }

    public T* Pointer => Tensor.Pointer;

    int IDeviceArray.ElementSize => sizeof(T);
    int IDeviceArray.Size => Shape.ArraySize;

    protected CudaNode(Shape shape) : base(shape)
    {
    }

    public abstract void EnsureHasUpdatedValues();

    public void Invalidate()
    {
        if (Outdated) return;

        foreach (var child in Dependents)
        {
            child.Invalidate();
        }

        Outdated = true;
    }

    // non-allocating transormations
    public CudaNode<T> Reshape(Shape shape) => new Reshape<T>(this, shape);
    public CudaNode<T> Transpose(Index[] axis) => new Permutation<T>(this, axis);
    public CudaNode<T> Transpose() => Transpose(Shape.GetTransposeAxis());

    public void CopyToHost(Span<T> destination)
    {
        EnsureHasUpdatedValues();
        Tensor.CopyToHost(destination);
    }

    public void CopyToDevice(ReadOnlySpan<T> source) => Tensor.CopyToDevice(source);

    public virtual void DisposeResources() {}

    public void Dispose()
    {
        DisposeResources();
    }

    public override string ToString()
    {
        var type = GetType().Name;
        return $"{type}, shape={Shape}";
    }
}
