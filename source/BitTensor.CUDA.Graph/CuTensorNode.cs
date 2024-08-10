using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode
{
    public delegate void ForwardFunction();
    public delegate CuTensorNode[] BackwardFunction(CuTensorNode grad, CuTensorNode self);

    public readonly CuContext Context;
    public readonly CuTensor Tensor;
    public readonly Shape Shape;
    public readonly ForwardFunction? Forward;
    public readonly BackwardFunction? Backward;
    public readonly CuTensorNode[] Children;
    public readonly List<CuTensorNode> Dependents;
    public bool Outdated;

    public CuTensorNode(CuTensor tensor)
    {
        Context = tensor.Context;
        Tensor = tensor;
        Shape = tensor.Shape;
        Children = [];
        Dependents = new(3);
        Outdated = false;
    }
    
    public CuTensorNode(CuTensor tensor, CuTensorNode[] children, ForwardFunction forward, BackwardFunction backward)
    {
        Context = tensor.Context;
        Tensor = tensor;
        Shape = tensor.Shape;
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

    public override int GetHashCode() => unchecked((int)Tensor.Id); // TODO: count nodes, not tensors

    public override string ToString() => $"Tensor #{Tensor.Id}, shape={Shape}";
}
