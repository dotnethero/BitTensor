using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Log<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> Input;
    internal readonly CuTensorBinaryPlan<T> Plan;
    internal readonly float Scale;

    public Log(CudaNode<T> input, float scale = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateAggregationPlan<T>(Operand.Log(Shape), Shape);
    }
    
    public override void Execute()
    {
        Plan.Execute(Input, Tensor, alpha: Scale, gamma: 0);
    }
    
    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        return [new Divide<T>(gradient, Input, Scale)];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}