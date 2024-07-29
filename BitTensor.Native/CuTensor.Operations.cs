using BitTensor.Abstractions;

namespace BitTensor.CUDA;

using Ops = GenericOperations<CuTensor, CuBackend>;

public partial class CuTensor
{
    public static CuTensor Sum(CuTensor a) => Ops.Sum(a);

    public static CuTensor Sum(CuTensor a, int[] axis) => Ops.Sum(a, axis);

    public static CuTensor MatMul(CuTensor a, CuTensor b)
    {
        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);

        return new CuTensor(
            a.Accelerator,
            [..batchDimensions, a.PrevDimension, b.LastDimension],
            children: [a, b],
            forward: static self => CuBackend.ExecuteMatMul(self.A, self.B, self),
            backward: Ops.NotSupported);
    }

    public void ApplyOffset(CuTensor offset) => CuBackend.ExecuteAdd(this, offset, this);

    public void ApplyScale(CuTensor scale) => CuBackend.ExecuteMultiply(this, scale, this);
}
