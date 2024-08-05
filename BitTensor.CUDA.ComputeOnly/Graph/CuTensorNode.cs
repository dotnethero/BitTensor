// ReSharper disable ConvertToPrimaryConstructor

namespace BitTensor.CUDA.ComputeOnly.Graph;

public partial class CuTensorNode : IDisposable
{
    public delegate void ForwardFunction();
    public delegate CuTensor[] BackwardFunction(CuTensor grad);

    internal readonly CuTensor Tensor;
    internal readonly CuTensorNode[] Children;
    internal readonly ForwardFunction? Forward;
    internal readonly BackwardFunction? Backward;

    internal bool Outdated;

    public CuTensorNode(CuTensor tensor)
    {
        Tensor = tensor;
        Children = [];
        Outdated = false;
    }
    
    public CuTensorNode(CuTensor tensor, CuTensorNode[] children, ForwardFunction forward, BackwardFunction backward)
    {
        Tensor = tensor;
        Children = children;
        Forward = forward;
        Backward = backward;
        Outdated = true;
    }

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

    public void Dispose()
    {
        // TODO release managed resources here
    }
}