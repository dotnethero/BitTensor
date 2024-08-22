using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Gemm<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<T> Product;
    internal readonly CudaNode<T> Bias;
    internal readonly CuTensorBroadcastPlan<T> Broadcast;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b, AbstractTensor c)
    {
        var left = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var shape = Shapes.Broadcast(left, c.Shape);
        return shape;
    }

    public Gemm(CudaNode<T> a, CudaNode<T> b, CudaNode<T> c) : base(GetShape(a, b, c), [a, b, c])
    {
        Bias = c;
        Product = new MatMul<T>(a, b);
        Broadcast = Context.cuTENSOR.CreateBroadcastPlan<T>(c.Shape, Shape);
    }

    public override void Execute()
    {
        Product.ExecuteInto(Tensor);
        Broadcast.Execute(Bias, Tensor, alpha: 1, gamma: 1f); // add inplace
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var productGrads = Product.Propagate(gradient);
        var biasAxis = Shapes.GetBroadcastedAxis(Bias.Shape, gradient.Shape);
        var biasGrad = Ops.Sum(gradient, axis: biasAxis).Reshape(Bias.Shape);
        return [..productGrads, biasGrad];
    }

    public override void DisposeResources()
    {
        Product.Dispose();
        Broadcast.Dispose();
    }
}