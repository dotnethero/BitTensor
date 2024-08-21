using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Exp<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudaNode<T> Input;
    internal readonly CuTensorBinaryPlan<T> Plan;
    internal readonly float Scale;

    public Exp(CudaNode<T> input, float scale = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateAggregationPlan<T>(Operand.Exp(Shape), Shape);
    }
    
    public override void Execute()
    {
        Plan.Execute(Input, Tensor, alpha: Scale, gamma: 0);
    }
    
    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        return [gradient * this];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}