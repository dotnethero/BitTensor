using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Broadcast<T> : CudaOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> A;
    internal readonly HashSet<Index> BroadcastedAxis;
    internal readonly float Scale;
    internal readonly CuTensorBroadcastPlan<T> Plan;

    public Broadcast(
        CudaNode<T> a,
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
    
    public override void Execute()
    {
        Plan.Execute(A, Tensor, alpha: Scale, gamma: 0);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        return [Ops.Sum(gradient, BroadcastedAxis, scale: Scale)];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}