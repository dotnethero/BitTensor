using System.Runtime.CompilerServices;

namespace BitTensor.CUDA.Graph;

public partial class CuTensorNode : IDisposable
{
    public delegate void ForwardFunction();
    public delegate CuTensor[] BackwardFunction(CuTensor grad);

    public readonly CuTensor Tensor;
    public readonly ForwardFunction? Forward;
    public readonly BackwardFunction? Backward;
    public readonly CuTensorNode[] Children;
    public readonly List<CuTensorNode> Dependents;
    public bool Outdated;

    public CuTensorNode(CuTensor tensor)
    {
        Tensor = tensor;
        Children = [];
        Dependents = new(3);
        Outdated = false;
    }
    
    public CuTensorNode(CuTensor tensor, CuTensorNode[] children, ForwardFunction forward, BackwardFunction backward)
    {
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
        foreach (var child in Dependents)
        {
            child.Invalidate();
        }

        Outdated = true;
    }

    public override int GetHashCode() => unchecked((int)Tensor.Id);

    public void Dispose()
    {
        // TODO release managed resources here
    }
}
