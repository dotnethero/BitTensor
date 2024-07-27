using BitTensor.Abstractions;

namespace BitTensor.CUDA;

using Ops = GenericOperations<CuTensor, CuBackend>;

public partial class CuTensor
{
    public static CuTensor Sum(CuTensor a) => Ops.Sum(a);
}
