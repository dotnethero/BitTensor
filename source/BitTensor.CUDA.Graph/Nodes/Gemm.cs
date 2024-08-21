using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class Gemm<T> : CudaOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<T> Product;
    internal readonly CudaNode<T> Bias;
    internal readonly CuTensorBroadcastPlan<T> Broadcast;
    internal readonly IEpilogue<T>? Epilogue;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b, AbstractTensor c)
    {
        var left = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var shape = Shapes.Broadcast(left, c.Shape);
        return shape;
    }

    public Gemm(CudaNode<T> a, CudaNode<T> b, CudaNode<T> c, IEpilogue<T>? epilogue = null) : base(GetShape(a, b, c), [a, b, c])
    {
        Bias = c;
        Product = new MatMul<T>(a, b);
        Broadcast = Context.cuTENSOR.CreateBroadcastPlan<T>(c.Shape, Shape);
        Epilogue = epilogue;
    }

    public override void Execute()
    {
        Product.ExecuteInto(Tensor);
        Broadcast.Execute(Bias, Tensor, alpha: 1, gamma: 1f); // add inplace
        Epilogue?.ExecuteInplace(Tensor);
    }

    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var epilogueGrad = Epilogue?.GetGradient(gradient) ?? gradient;
        var productGrads = Product.Propagate(epilogueGrad);
        var biasAxis = Shapes.GetBroadcastedAxis(Bias.Shape, epilogueGrad.Shape);
        var biasGrad = Ops.Sum(gradient, axis: biasAxis).Reshape(Bias.Shape);
        return [..productGrads, biasGrad];
    }

    public override void DisposeResources()
    {
        Product.Dispose();
        Broadcast.Dispose();
    }
}