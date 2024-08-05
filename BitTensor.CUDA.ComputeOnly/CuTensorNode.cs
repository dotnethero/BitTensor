// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly;

public class CuTensorNode : IDisposable
{
    public delegate void ForwardFunction(CuTensor self);
    public delegate CuTensor[] BackwardFunction(CuTensor self, CuTensor grad);

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

        Forward?.Invoke(Tensor);
        Outdated = false;
    }

    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Tensor.Shape, b.Tensor.Shape);
        var output = new CuTensor(shape);
        return new CuTensorNode(
            output, 
            children: [a, b],
            forward: (self) => CuTensor.Add(a.Tensor, b.Tensor, output),
            backward: (self, grad) => throw new NotImplementedException());
    }

    public void Dispose()
    {
        // TODO release managed resources here
    }
}