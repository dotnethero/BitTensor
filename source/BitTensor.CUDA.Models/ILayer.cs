using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer
{
    CuTensorWeight[] Parameters { get; }
    CuTensorNode Compute(CuTensorNode input);
}

