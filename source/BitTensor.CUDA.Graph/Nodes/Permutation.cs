using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Permutation<T> : AbstractTransformation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly Index[] InversePermutation;

    private static Shape GetShape(AbstractTensor source, Index[] axis) => source.Shape.Transpose(axis);

    public Permutation(
        CudaNode<T> source,
        Index[] axis) : 
        base(GetShape(source, axis), source, t => t.Transpose(axis)) =>
        InversePermutation = Axis.InvertPermutation(axis);

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient) => 
        [gradient.Transpose(InversePermutation)];
}
