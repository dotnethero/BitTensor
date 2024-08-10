using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer
{
    CuTensorWeights[] Parameters { get; }
    CuTensorNode Compute(CuTensorNode input);
}

