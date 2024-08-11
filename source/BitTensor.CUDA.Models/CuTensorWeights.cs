using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public class CuTensorWeights : CuTensorNode
{
    private readonly CuTensorBinaryPlan<float> _plan;

    public CuTensorWeights(CuTensor tensor) : base(tensor)
    {
        _plan = Context.CreateAggregationPlan<float>(tensor);
    }

    public void AdjustWeights(CuTensor gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}