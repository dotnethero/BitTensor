using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class GemmReluCudnn<T> : AbstractOperation<float> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<float> Product;
    internal readonly CudaNode<float> Bias;
    internal readonly ReLU ReLU;

    internal readonly CudnnContext CudnnContext;
    internal readonly ICudnnGraph CudnnGraph;
    internal readonly ICudnnPlan CudnnPlan;
    internal readonly Lazy<CudnnVariantPack<float>> CudnnPack;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b, AbstractTensor c)
    {
        var left = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var shape = Shapes.Broadcast(left, c.Shape);
        return shape;
    }

    public GemmReluCudnn(CudaNode<float> a, CudaNode<float> b, CudaNode<float> c, float alpha) : base(GetShape(a, b, c), [a, b, c])
    {
        // TODO: add own propagate function

        Bias = c;
        Product = new MatMul<float>(a, b);
        ReLU = new ReLU(Product + Bias);

        CudnnContext = new CudnnContext();
        CudnnGraph = new CudnnGemmReluGraph<float>(CudnnContext, a, b, c, this, alpha);
        CudnnPack = new Lazy<CudnnVariantPack<float>>(() => new([a, b, c, this]));
        CudnnPlan = CudnnGraph.GetExecutionPlan();
    }

    public override void Execute()
    {
        CudnnPlan.Execute(CudnnPack.Value);
    }

    public override CudaNode<float>[] Propagate(CudaNode<float> gradient)
    {
        var g = ReLU.Propagate(gradient)[0];
        var productGrads = Product.Propagate(g);
        var biasAxis = Shapes.GetBroadcastedAxis(Bias.Shape, g.Shape);
        var biasGrad = Ops.Sum(g, axis: biasAxis).Reshape(Bias.Shape);
        return [..productGrads, biasGrad];
    }

    public override void DisposeResources()
    {
        CudnnPack.Value.Dispose();
        CudnnPlan.Dispose();
        CudnnGraph.Dispose();
        CudnnContext.Dispose();
    }
}