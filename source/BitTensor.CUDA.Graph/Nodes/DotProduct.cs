using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class DotProduct<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly AbstractNode<T> B;
    internal readonly float Scale;
    internal readonly CuTensorContractionPlan<T> Plan;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b) => Shapes.Broadcast(a.Shape, b.Shape);

    public DotProduct(AbstractNode<T> a, AbstractNode<T> b, float scale = 1f) : base(GetShape(a, b), [a, b])
    {
        A = a;
        B = b;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateContractionPlan<T>(a.Shape, b.Shape, Shape);
    }

    public override void EnsureHasUpdatedValue()
    {
        A.EnsureHasUpdatedValue();
        B.EnsureHasUpdatedValue();
        Plan.Execute(A, B, Tensor, alpha: Scale, beta: 0f);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient) =>
    [
        gradient * B, 
        gradient * A
    ];
}