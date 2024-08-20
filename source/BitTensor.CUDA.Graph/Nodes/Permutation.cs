using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Permutation<T> : AbstractTransformation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Index[] InversePermutation;

    private static Shape GetShape(AbstractTensor source, Index[] axis) => source.Shape.Transpose(axis);

    public Permutation(
        AbstractNode<T> source,
        Index[] axis) : 
        base(GetShape(source, axis), source, t => t.Transpose(axis)) =>
        InversePermutation = Axis.InvertPermutation(axis);

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient) => 
        [gradient.Transpose(InversePermutation)];

    public override void EnsureHasUpdatedValue() => Source.EnsureHasUpdatedValue();
}
