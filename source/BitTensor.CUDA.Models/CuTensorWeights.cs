using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public class CuTensorWeights : CuTensorNode
{
    private readonly CuTensorBinaryPlan _plan;

    public CuTensorWeights(CuTensor tensor) : base(tensor)
    {
        _plan = Context.CreateAggregationPlan(tensor);
    }

    public void AdjustWeights(CuTensor gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}