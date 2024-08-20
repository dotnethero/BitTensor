using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Broadcast<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly HashSet<Index> BroadcastedAxis;
    internal readonly float Scale;
    internal readonly CuTensorBroadcastPlan<T> Plan;

    public Broadcast(
        AbstractNode<T> a,
        Shape shape,
        float scale = 1f) :
        base(shape, [a])
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        A = a;
        Scale = scale;
        BroadcastedAxis = Shapes.GetBroadcastedAxis(a.Shape, shape);
        Plan = Context.cuTENSOR.CreateBroadcastPlan<T>(a.Shape, shape);
    }
    
    public override void Execute(CudaTensor<T> output)
    {
        Plan.Execute(A, output, alpha: Scale, gamma: 0);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        return [new Sum<T>(gradient, BroadcastedAxis, scale: Scale)];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}