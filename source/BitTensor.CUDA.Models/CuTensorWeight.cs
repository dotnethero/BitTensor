using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Models;

public class CuTensorWeight : CuTensorNode
{
    private readonly CuTensorOffsetPlan _plan;

    public CuTensorWeight(CuTensorContext context, CuTensor tensor) : base(context, tensor, true)
    {
        _plan = new CuTensorOffsetPlan(context, tensor, tensor);
    }

    public void AdjustWeights(CuTensor gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}