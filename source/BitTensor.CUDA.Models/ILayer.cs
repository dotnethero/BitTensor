using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public interface ILayer
{
    CuTensorNode[] Parameters { get; }
    CuTensorNode Compute(CuTensorNode input);
}

