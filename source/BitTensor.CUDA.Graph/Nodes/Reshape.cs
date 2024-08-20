using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Reshape<T> : AbstractTransformation<T> where T : unmanaged, IFloatingPoint<T>
{
    private static Shape GetShape(AbstractTensor source, Shape shape)
    {
        source.Shape.EnsureCanReshape(shape);
        return shape;
    }

    public Reshape(
        AbstractNode<T> source,
        Shape shape) : 
        base(GetShape(source, shape), source, t => t.Reshape(shape)) { }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient) => 
        [gradient.Reshape(Source.Shape)];

    public override void EnsureHasUpdatedValue() => Source.EnsureHasUpdatedValue();
}
