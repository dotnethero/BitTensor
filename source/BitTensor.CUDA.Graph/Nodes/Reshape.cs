using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Reshape<T> : AbstractTransformation<T> where T : unmanaged, IFloatingPoint<T>
{
    private static Shape GetShape(AbstractTensor source, Shape shape)
    {
        source.Shape.EnsureCanReshape(shape);
        return shape;
    }

    public Reshape(
        CudaNode<T> source,
        Shape shape) : 
        base(GetShape(source, shape), source, t => t.Reshape(shape)) { }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient) => 
        [gradient.Reshape(Source.Shape)];
}
