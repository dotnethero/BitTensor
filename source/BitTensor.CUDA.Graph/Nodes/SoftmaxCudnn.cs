using System.Numerics;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph.Nodes;

internal sealed class SoftmaxCudnn<T> : AbstractOperation<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CudnnContext CudnnContext;
    internal readonly ICudnnGraph CudnnGraph;
    internal readonly ICudnnPlan CudnnPlan;
    internal readonly Lazy<CudnnVariantPack<T>> CudnnPack;
    internal readonly HashSet<Index> Axis;
    
    public SoftmaxCudnn(CudaNode<T> input, HashSet<Index> axis) : base(input.Shape, [input])
    {
        Axis = axis;
        CudnnContext = new CudnnContext();
        CudnnGraph = new CudnnSoftmaxGraph<T>(CudnnContext, input, axis, this);
        CudnnPack = new(() => new([input, this]));
        CudnnPlan = CudnnGraph.GetExecutionPlan();
    }

    public override void Execute()
    {
        CudnnPlan.Execute(CudnnPack.Value);
    }
    
    public override CudaNode<T>[] Propagate(CudaNode<T> gradient)
    {
        var sum = Ops.Sum(this * gradient, Axis, keepDims: true);
        var result = this * (gradient - sum);
        return [result];
    }

    public override void DisposeResources()
    {
        CudnnPack.Value.Dispose();
        CudnnPlan.Dispose();
        CudnnGraph.Dispose();
        CudnnContext.Dispose();
    }
}