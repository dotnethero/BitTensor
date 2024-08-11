using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode : AbstractTensor
{
    public delegate void ForwardFunction();
    public delegate CuTensorNode[] BackwardFunction(CuTensorNode grad, CuTensorNode self);

    public readonly CuContext Context;
    public readonly CuTensor Tensor;
    public readonly ForwardFunction? Forward;
    public readonly BackwardFunction? Backward;
    public readonly CuTensorNode[] Children;
    public readonly List<CuTensorNode> Dependents;
    public bool Outdated;

    public CuTensorNode(CuTensor tensor) : base(tensor.Shape)
    {
        Context = tensor.Context;
        Tensor = tensor;
        Children = [];
        Dependents = new(3);
        Outdated = false;
    }
    
    public CuTensorNode(CuTensor tensor, CuTensorNode[] children, ForwardFunction forward, BackwardFunction backward) : base(tensor.Shape)
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

    public override int GetHashCode() => unchecked((int)Id);

    public override string ToString() => $"Tensor #{Id}, shape={Shape}";
}
