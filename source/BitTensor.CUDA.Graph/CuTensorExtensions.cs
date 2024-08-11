using BitTensor.CUDA.Graph;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class CuTensorExtensions
{
    public static CuTensorNode AsNode(this CuTensor tensor) => new(tensor);
}
