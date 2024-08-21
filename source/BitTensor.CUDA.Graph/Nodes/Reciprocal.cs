using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Reciprocal<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly AbstractNode<T> Input;
    internal readonly CuTensorBinaryPlan<T> Plan;
    internal readonly float Scale;

    public Reciprocal(AbstractNode<T> input, float scale = 1f) : base(input.Shape, [input])
    {
        Input = input;
        Scale = scale;
        Plan = Context.cuTENSOR.CreateAggregationPlan<T>(Operand.Rcp(Shape), Shape);
    }
    
    public override void EnsureHasUpdatedValue()
    {
        Input.EnsureHasUpdatedValue();
        Plan.Execute(Input, Tensor, alpha: Scale, gamma: 0);
    }
    
    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var square = new Multiply<T>(this, this, -Scale); // TODO: Simplify square
        var result = new Multiply<T>(gradient, square);
        return [result];
    }

    public override void DisposeResources()
    {
        Plan.Dispose();
    }
}