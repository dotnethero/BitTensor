using System.Numerics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public class CuWeights<T> : CuNode<T> where T : unmanaged, IFloatingPoint<T>
{
    private readonly CuTensorBinaryPlan<T> _plan;

    public CuWeights(CuContext context, CuTensor<T> tensor) : base(context, tensor)
    {
        _plan = Context.cuTENSOR.CreateAggregationPlan<T>(tensor);
    }

    public void AdjustWeights(CuTensor<T> gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}