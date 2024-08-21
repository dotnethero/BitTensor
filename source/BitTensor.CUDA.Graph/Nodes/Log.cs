using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Log<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> Input;
    internal readonly CuTensorBinaryPlan<T> Plan;
    internal readonly float Scale;

    public Log(AbstractNode<T> input, float scale = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateAggregationPlan<T>(Operand.Log(Shape), Shape);
    }
    
    public override void EnsureHasUpdatedValue()
    {
        Input.EnsureHasUpdatedValue();
        Plan.Execute(Input, Tensor, alpha: Scale, gamma: 0);
    }
    
    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var reciprocal = new Reciprocal<T>(this);
        var result = new Multiply<T>(gradient, reciprocal, Scale); // TODO: Scaled division
        return [result];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}