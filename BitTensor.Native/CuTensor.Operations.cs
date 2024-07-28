using BitTensor.Abstractions;

namespace BitTensor.CUDA;

using Ops = GenericOperations<CuTensor, CuBackend>;

public partial class CuTensor
{
    public static CuTensor Sum(CuTensor a) => Ops.Sum(a);

    public static CuTensor Sum(CuTensor a, int[] axis) => Ops.Sum(a, axis);

    public void ApplyOffset(CuTensor offset) => CuBackend.ExecuteAdd(this, offset, this);

    public void ApplyScale(CuTensor scale) => CuBackend.ExecuteMultiply(this, scale, this);
}
