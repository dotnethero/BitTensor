using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class GemmCudnn<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly MatMul<T> Product;
    internal readonly CudaNode<T> Bias;

    internal readonly CudnnContext CudnnContext;
    internal readonly Lazy<CudnnCompiledGraph<T>> CudnnGraph;

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
        CudnnGraph = new Lazy<CudnnCompiledGraph<T>>(() => Compile(a, b, c));
    }


    private CudnnCompiledGraph<T> Compile(CudaNode<T> a, CudaNode<T> b, CudaNode<T> c)
    {
        var ta = new CudnnTensorDescriptor<float>(a.Tensor);
        var tb = new CudnnTensorDescriptor<float>(b.Tensor);
        var tc = new CudnnTensorDescriptor<float>(c.Tensor);
        var tt = new CudnnTensorDescriptor<float>(Shape, -1, isVirtual: true);
        var to = new CudnnTensorDescriptor<float>(Tensor);

        var mmc = new CudnnMatMulOperator<float>();
        var mm = new CudnnMatMulOperation<float>(mmc, ta, tb, tt);

        var pwc = new CudnnPointwiseOperator<float>();
        var pw = new CudnnPointwiseOperation<float>(pwc, tt, tc, to);

        var graph = new CudnnGraph(CudnnContext, [mm, pw]);
        var heuristics = new CudnnEngineHeuristics(graph);

        var plan = new CudnnExecutionPlan(CudnnContext, heuristics.GetConfiguration());
        var pack = new CudnnVariantPack<T>([a.Tensor, b.Tensor, c.Tensor, Tensor]);

        return new CudnnCompiledGraph<T>(plan, pack);
    }

    public override void Execute()
    {
        var graph = CudnnGraph.Value;
        CudnnContext.ExecutePlan(
            graph.Plan,
            graph.Pack);
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
        CudnnContext.Dispose();

        // TODO: dispose cuDNN wrappers
    }
}