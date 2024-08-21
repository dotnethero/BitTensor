using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph.Epilogues;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

public sealed class Gemm<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<T> Product;
    internal readonly AbstractNode<T> Bias;
    internal readonly CuTensorBroadcastPlan<T> Broadcast;
    internal readonly IEpilogue<T>? Epilogue;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b, AbstractTensor c)
    {
        var left = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var shape = Shapes.Broadcast(left, c.Shape);
        return shape;
    }

    public Gemm(AbstractNode<T> a, AbstractNode<T> b, AbstractNode<T> c, IEpilogue<T>? epilogue = null) : base(GetShape(a, b, c), [a, b, c])
    {
        Bias = c;
        Product = new MatMul<T>(a, b);
        Broadcast = Context.cuTENSOR.CreateBroadcastPlan<T>(c.Shape, Shape);
        Epilogue = epilogue;
    }

    public override void EnsureHasUpdatedValue()
    {
        Bias.EnsureHasUpdatedValue();
        Product.ExecuteInto(Tensor);
        Broadcast.Execute(Bias, Tensor, alpha: 1, gamma: 1f); // add inplace
        Epilogue?.ExecuteInplace(Tensor);
    }

    public override AbstractNode<T>[] Propagate(AbstractNode<T> gradient)
    {
        var epilogueGrad = Epilogue?.GetGradient(gradient) ?? gradient;
        var productGrads = Product.Propagate(epilogueGrad);
        var biasAxis = Shapes.GetBroadcastedAxis(Bias.Shape, epilogueGrad.Shape);
        var biasGrad = new Sum<T>(gradient, axis: biasAxis).Reshape(Bias.Shape);
        return [..productGrads, biasGrad];
    }

    public override void DisposeResources()
    {
        Product.Dispose();
        Broadcast.Dispose();
    }
}