using BitTensor.CUDA.Graph;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class CuTensorExtensions
{
    public static CuTensorNode ToNode(this CuTensor tensor) => new(tensor, owned: true);
}
