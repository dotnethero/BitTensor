using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Wrappers;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class CuTensorExtensions
{
    public static CuTensorNode CreateNode(this CuTensor tensor, CuTensorContext context) => new(context, tensor, owned: true);
}
