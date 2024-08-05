// ReSharper disable ConvertToPrimaryConstructor

using BitTensor.Abstractions;

namespace BitTensor.CUDA.ComputeOnly;

public class CuTensorNode : IDisposable
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

    public static CuTensorNode operator +(CuTensorNode a, CuTensorNode b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Tensor.Shape, b.Tensor.Shape);
        var output = new CuTensor(shape);
        return new CuTensorNode(
            output, 
            children: [a, b],
            forward: () => CuTensor.Add(a.Tensor, b.Tensor, output),
            backward: (grad) =>
            {
                var adims = Shapes.GetBroadcastedAxis(a.Tensor.Shape, output.Shape);
                var bdims = Shapes.GetBroadcastedAxis(b.Tensor.Shape, output.Shape);
                var agrad = CuTensor.Sum(grad, axis: adims);
                var bgrad = CuTensor.Sum(grad, axis: bdims);
                return
                [
                    CuTensor.Reshape(agrad, a.Tensor.Shape),
                    CuTensor.Reshape(bgrad, b.Tensor.Shape),
                ];
            });
    }

    public void Dispose()
    {
        // TODO release managed resources here
    }
}