using System.Numerics;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

using OpCode = cutensorOperator_t;

public class Sum<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    Reduction<T>(a, axis, OpCode.CUTENSOR_OP_ADD, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

public class Max<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    Reduction<T>(a, axis, OpCode.CUTENSOR_OP_MAX, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

public class Min<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    Reduction<T>(a, axis, OpCode.CUTENSOR_OP_MIN, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

public abstract class Reduction<T> : AbstractNode<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> A;
    internal readonly float Scale;
    internal readonly CuTensorReductionPlan<T> Plan;

    protected Reduction(
        AbstractNode<T> a,
        HashSet<Index> axis,
        OpCode operation,
        float scale = 1f,
        bool keepDims = false) :
        base(a.Shape.Reduce(axis, keepDims), [a])
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