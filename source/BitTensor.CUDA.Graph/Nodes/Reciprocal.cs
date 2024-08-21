using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Reciprocal<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> Input;
    internal readonly CuTensorBinaryPlan<T> Plan;
    internal readonly float Scale;

    public Reciprocal(CudaNode<T> input, float scale = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateAggregationPlan<T>(Operand.Rcp(Shape), Shape);
    }
    
    public override void Execute()
    {
        Plan.Execute(Input, Tensor, alpha: Scale, gamma: 0);
    }
    
    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var square = new Multiply<T>(this, this, -Scale); // TODO: Simplify square
        return [gradient * square];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}