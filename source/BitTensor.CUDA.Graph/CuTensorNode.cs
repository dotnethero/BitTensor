using System.Numerics;
using System.Runtime.CompilerServices;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode<T> : AbstractTensor, IDeviceArray<T>, IHasContext where T : unmanaged, INumberBase<T>
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
    
    public unsafe T* Pointer => Tensor.Pointer;

    int IDeviceArray<T>.ElementSize => Tensor.Array.ElementSize;
    long IDeviceArray<T>.Size => Tensor.Array.Size;

    CuContext IHasContext.GetContext() => Context;

    public CuTensorNode(CuTensor<T> tensor) : base(tensor.Shape)
    {
        Context = tensor.Context;
        Tensor = tensor;
        Children = [];
        Dependents = new(3);
        Outdated = false;
    }
    
    public CuTensorNode(CuTensor<T> tensor, CuTensorNode<T>[] children, ForwardFunction forward, BackwardFunction backward) : base(tensor.Shape)
    {
        Context = tensor.Context;
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

        cudaRT.cudaDeviceSynchronize();
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

    public override int GetHashCode() => unchecked((int)Id);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
