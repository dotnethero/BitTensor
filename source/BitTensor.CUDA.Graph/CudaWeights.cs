using System.Numerics;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Graph;

public sealed class CudaWeights<T> : CudaVariable<T> where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorBinaryPlan<T> AggregationPlan;

    public CudaWeights(CudaContext context, CudaTensor<T> tensor) : base(context, tensor)
    {
        AggregationPlan = Context.cuTENSOR.CreateAggregationPlan<T>(tensor.Shape);
    }

    public void AdjustWeights(CudaTensor<T> gradient, float lr)
    {
        AggregationPlan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }

    public override void DisposeResources()
    {
        AggregationPlan.Dispose();
    }
}