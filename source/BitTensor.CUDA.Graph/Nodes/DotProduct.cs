using System.Numerics;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class DotProduct<T> : CudaOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> A;
    internal readonly CudaNode<T> B;
    internal readonly float Scale;
    internal readonly CuTensorContractionPlan<T> Plan;

    public DotProduct(CudaNode<T> a, CudaNode<T> b, float scale = 1f) : base([], [a, b])
    {
        A = a;
        B = b;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateContractionPlan<T>(a.Shape, b.Shape, Shape);
    }

    public override void Execute()
    {
        Plan.Execute(A, B, Tensor, alpha: Scale, beta: 0f);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient) =>
    [
        gradient * B, 
        gradient * A
    ];
}