using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal abstract class AbstractReduction<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> Source;
    internal readonly CuTensorReductionPlan<T> Plan;
    internal readonly float Scale;

    private static Shape GetShape(
        AbstractTensor a,
        HashSet<Index> axis,
        bool keepDims) => 
        a.Shape.Reduce(axis, keepDims);

    protected AbstractReduction(
        CudaNode<T> source,
        HashSet<Index> axis,
        cutensorOperator_t operation,
        float scale = 1f,
        bool keepDims = false) :
        base(GetShape(source, axis, keepDims), [source])
    {
        Source = source;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateReductionPlan<T>(source.Shape, Shape, axis, operation, keepDims);
    }

    public override void Execute()
    {
        Plan.Execute(Source, Tensor, Scale);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        return [new Broadcast<T>(gradient, Source.Shape, Scale)];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}