using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class GemmCudnn<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<T> Product;
    internal readonly CudaNode<T> Bias;

    internal readonly CudnnContext CudnnContext;
    internal readonly CudnnExecutionPlan CudnnPlan;
    internal readonly Lazy<CudnnVariantPack<T>> CudnnPack;

    private static Shape GetShape(AbstractTensor a, AbstractTensor b, AbstractTensor c)
    {
        var left = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var shape = Shapes.Broadcast(left, c.Shape);
        return shape;
    }

    public GemmCudnn(CudaNode<T> a, CudaNode<T> b, CudaNode<T> c) : base(GetShape(a, b, c), [a, b, c])
    {
        Bias = c;
        Product = new MatMul<T>(a, b);

        CudnnContext = new CudnnContext();

        var ta = new CudnnTensorDescriptor<float>(a);
        var tb = new CudnnTensorDescriptor<float>(b);
        var tc = new CudnnTensorDescriptor<float>(c);
        var tt = new CudnnTensorDescriptor<float>(Shape, -1, isVirtual: true);
        var to = new CudnnTensorDescriptor<float>(this);

        var mmc = new CudnnMatMulOperator<float>();
        var mm = new CudnnMatMulOperation<float>(mmc, ta, tb, tt);

        var pwc = new CudnnPointwiseOperator<float>();
        var pw = new CudnnPointwiseOperation<float>(pwc, tt, tc, to);

        var graph = new CudnnGraph(CudnnContext, [mm, pw]);
        var heuristics = new CudnnEngineHeuristics(graph);

        CudnnPlan = new CudnnExecutionPlan(CudnnContext, heuristics.GetConfiguration());
        CudnnPack = new Lazy<CudnnVariantPack<T>>(() => new([a, b, c, this]));
    }

    public override void Execute()
    {
        CudnnContext.ExecutePlan(
            CudnnPlan,
            CudnnPack.Value);
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
        CudnnPack.Value.Dispose();
        CudnnPlan.Dispose();
        CudnnContext.Dispose();
        
        // TODO: dispose other cuDNN wrappers
    }
}