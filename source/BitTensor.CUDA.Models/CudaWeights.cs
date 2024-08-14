using System.Numerics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public sealed class CudaWeights<T> : CudaNode<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly CuTensorBinaryPlan<T> _plan;

    public CudaWeights(CudaContext context, CudaTensor<T> tensor) : base(context, tensor)
    {
        _plan = Context.cuTENSOR.CreateAggregationPlan<T>(tensor.Shape);
    }

    public void AdjustWeights(CudaTensor<T> gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}