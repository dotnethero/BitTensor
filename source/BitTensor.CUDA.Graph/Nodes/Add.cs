using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Add<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> A;
    internal readonly CudaNode<T> B;
    internal readonly float Alpha;
    internal readonly float Beta;
    internal readonly CuTensorTernaryPlan<T> Plan;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b) => Shapes.Broadcast(a.Shape, b.Shape);

    public Add(CudaNode<T> a, CudaNode<T> b, float alpha = 1f, float beta = 1f) : base(GetShape(a, b), [a, b])
    {
        A = a;
        B = b;
        Alpha = alpha;
        Beta = beta;
        Plan = Context.cuTENSOR.CreateAddPlan<T>(a.Shape, b.Shape, Shape);
    }

    public override void Execute()
    {
        Plan.Execute(A, B, Tensor, Alpha, Beta, gamma: 0f);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var adims = Shapes.GetBroadcastedAxis(A.Shape, gradient.Shape);
        var bdims = Shapes.GetBroadcastedAxis(B.Shape, gradient.Shape);
        return
        [
            Ops.Sum(gradient, axis: adims, scale: Alpha).Reshape(A.Shape),
            Ops.Sum(gradient, axis: bdims, scale: Beta).Reshape(B.Shape)
        ];
    }
}