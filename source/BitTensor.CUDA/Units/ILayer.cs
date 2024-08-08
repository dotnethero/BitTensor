using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Units;

public interface ILayer
{
    CuTensorNode[] Parameters { get; }
    CuTensorNode Compute(CuTensorNode input);
}

