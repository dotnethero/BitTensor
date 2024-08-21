using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Multiply<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly AbstractNode<T> B;
    internal readonly float Scale;
    internal readonly CuTensorTernaryPlan<T> Plan;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b) => Shapes.Broadcast(a.Shape, b.Shape);

    public Multiply(AbstractNode<T> a, AbstractNode<T> b, float scale = 1f) : base(GetShape(a, b), [a, b])
    {
        A = a;
        B = b;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateMultiplyPlan<T>(a.Shape, b.Shape, Shape);
    }

    public override void EnsureHasUpdatedValue()
    {
        A.EnsureHasUpdatedValue();
        B.EnsureHasUpdatedValue();
        Plan.Execute(A, B, Tensor, alpha: Scale, beta: 1f, gamma: 0f);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var agrad = new Multiply<T>(gradient, B, Scale);
        var bgrad = new Multiply<T>(gradient, A, Scale);
        var adims = Shapes.GetBroadcastedAxis(A.Shape, agrad.Shape);
        var bdims = Shapes.GetBroadcastedAxis(B.Shape, bgrad.Shape);
        return
        [
            Ops.Sum(agrad, axis: adims).Reshape(A.Shape),
            Ops.Sum(bgrad, axis: bdims).Reshape(B.Shape)
        ];
    }
}