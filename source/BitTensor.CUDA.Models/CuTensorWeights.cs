using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Models;

public class CuTensorWeights : CuTensorNode
{
    private readonly CuTensorOffsetPlan _plan;

    public CuTensorWeights(CuTensor tensor) : base(tensor)
    {
        _plan = new CuTensorOffsetPlan(Context.cuTENSOR, tensor, tensor);
    }

    public void AdjustWeights(CuTensor gradient, float lr)
    {
        _plan.Execute(gradient, Tensor, -lr);
        Invalidate();
    }
}