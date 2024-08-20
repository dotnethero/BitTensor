using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public abstract class AbstractReduction<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly float Scale;
    internal readonly CuTensorReductionPlan<T> Plan;
    
    private static Shape GetShape(
        AbstractTensor a,
        HashSet<Index> axis,
        bool keepDims) => 
        a.Shape.Reduce(axis, keepDims);

    protected AbstractReduction(
        AbstractNode<T> a,
        HashSet<Index> axis,
        cutensorOperator_t operation,
        float scale = 1f,
        bool keepDims = false) :
        base(GetShape(a, axis, keepDims), [a])
    {
        A = a;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateReductionPlan<T>(a.Shape, Shape, axis, operation, keepDims);
    }

    public override void Execute(CudaTensor<T> output)
    {
        Plan.Execute(A, output, Scale);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        return [new Broadcast<T>(gradient, A.Shape, Scale)];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}