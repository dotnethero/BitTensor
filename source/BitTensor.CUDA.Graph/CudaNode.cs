using System.Numerics;
using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public unsafe partial class CudaNode<T> : AbstractTensor, IDeviceArray<T> where T : unmanaged, IFloatingPoint<T>
{
    public delegate void ForwardFunction(CudaTensor<T> output);
    public delegate CudaNode<T>[] BackwardFunction(CudaNode<T> grad, CudaNode<T> self);

    public readonly CudaContext Context;
    public CudaTensor<T> Tensor => TensorGetter.Value;
    public readonly Lazy<CudaTensor<T>> TensorGetter;
    public readonly ForwardFunction? Forward;
    public readonly BackwardFunction? Backward;
    public readonly CudaNode<T>[] Children;
    public readonly List<CudaNode<T>> Dependents;
    public bool Outdated;
    
    // TODO: inline
    public T* Pointer => Tensor.Pointer;

    int IDeviceArray.ElementSize => sizeof(T);
    int IDeviceArray.Size => Shape.ArraySize;

    public CudaNode(CudaContext context, CudaTensor<T> tensor) : base(tensor.Shape)
    {
        Context = context;
        TensorGetter = new(tensor);
        Children = [];
        Dependents = new(2);
        Outdated = false;
    }
    
    public CudaNode(CudaContext context, Shape shape, CudaNode<T>[] children, ForwardFunction forward, BackwardFunction backward) : base(shape)
    {
        Context = context;
        TensorGetter = new(() => context.Allocate<T>(shape));
        Forward = forward;
        Backward = backward;
        Children = children;
        Dependents = new(2);
        Outdated = true;

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }
    
    private CudaNode(CudaContext context, Shape shape, CudaNode<T> source, Func<CudaTensor<T>, CudaTensor<T>> transformation, BackwardFunction backward) : base(shape)
    {
        Context = context;
        TensorGetter = new(() => transformation(source.Tensor));
        Forward = static _ => {};
        Backward = backward;
        Children = [source];
        Dependents = new(2);
        Outdated = true;

        foreach (var child in Children)
        {
            child.Dependents.Add(this);
        }
    }

    public CudaNode<T> Reshape(Shape shape) // no allocation
    {
        Shape.EnsureCanReshape(shape);
        return new(
            Context,
            shape,
            source: this,
            transformation: tensor => tensor.Reshape(shape),
            backward: (grad, _) => [grad.Reshape(Shape)]);
    }

    public CudaNode<T> Transpose(Index[] axis) // no allocation
    {
        var shape = Shape.Transpose(axis);
        var inverted = Axis.InvertPermutation(axis);
        return new(
            Context,
            shape,
            source: this,
            transformation: tensor => tensor.Transpose(axis),
            backward: (grad, _) => [grad.Transpose(inverted)]);
    }

    public CudaNode<T> Transpose() => Transpose(Shape.GetTransposeAxis());

    [MethodImpl(MethodImplOptions.NoInlining)]
    public void EnsureHasUpdatedValues()
    {
        if (!Outdated) return;

        foreach (var child in Children)
        {
            child.EnsureHasUpdatedValues();
        }

        Forward?.Invoke(Tensor);
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

    public void Dispose()
    {
        // TODO: release plans early
    }
}
