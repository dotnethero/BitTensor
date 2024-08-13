using System.Numerics;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public class CuTensorWeights<T> : CuTensorNode<T> where T : unmanaged, INumberBase<T>
{
    private readonly CuTensorBinaryPlan<T> _plan;

    public CuTensorWeights(CuContext context, CuTensor<T> tensor) : base(context, tensor)
    {
        _plan = Context.cuTENSOR.CreateAggregationPlan<T>(tensor);
    }

    public void AdjustWeights(CuTensor<T> gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}