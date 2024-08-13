using System.Numerics;
using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged, INumberBase<T>
{
    public delegate void ForwardFunction();
    public delegate CuTensorNode<T>[] BackwardFunction(CuTensorNode<T> grad, CuTensorNode<T> self);

    public readonly CuContext Context;
    public readonly CuTensor<T> Tensor;
    public readonly ForwardFunction? Forward;
    public readonly BackwardFunction? Backward;
    public readonly CuTensorNode<T>[] Children;
    public readonly List<CuTensorNode<T>> Dependents;
    public bool Outdated;
    
    // TODO: inline
    public unsafe T* Pointer => Tensor.Pointer;

    int IDeviceArray<T>.ElementSize => Tensor.Array.ElementSize;
    int IDeviceArray<T>.Size => Tensor.Array.Size;

    public CuTensorNode(CuContext context, CuTensor<T> tensor) : base(tensor.Shape)
    {
        Context = context;
        Tensor = tensor;
        Children = [];
        Dependents = new(3);
        Outdated = false;
    }
    
    public CuTensorNode(CuContext context, CuTensor<T> tensor, CuTensorNode<T>[] children, ForwardFunction forward, BackwardFunction backward) : base(tensor.Shape)
    {
        Context = context;
        Tensor = tensor;
        Forward = forward;
        Backward = backward;
        Children = children;
        Dependents = new(3);
        Outdated = true;

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        foreach (var child in Children)
        {
            child.EnsureHasUpdatedValues();
        }

        Forward?.Invoke();
        Outdated = false;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    public void Invalidate()
    {
        if (Outdated) return;

        foreach (var child in Dependents)
        {
            child.Invalidate();
        }

        Outdated = true;
    }

    public T[] CopyToHost()
    {
        EnsureHasUpdatedValues();
        IDeviceArray<T> tensor = Tensor;
        return tensor.CopyToHost();
    }

    public void CopyToHost(Span<T> destination) => Tensor.CopyToHost(destination);

    public void CopyToDevice(ReadOnlySpan<T> source) => Tensor.CopyToDevice(source);

    public void LoadBatches(Dataset<T> dataset, int[] batchIndexes)
    {
        var stride = dataset.Shape.Strides[0];

        for (var i = 0; i < batchIndexes.Length; i++)
        {
            var sampleIndex = batchIndexes[i];
            var batch = new ReadOnlySpan<T>(dataset.Data, sampleIndex * stride, stride);
            Tensor.Array.CopyToDevice(batch, i * stride, stride);
        }
    }

    public override int GetHashCode() => unchecked((int)Id);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
