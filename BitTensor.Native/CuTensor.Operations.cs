using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using Ops = GenericOperations<CuTensor, CuBackend>;

public partial class CuTensor
{
    public static CuTensor Transpose(CuTensor a) => Transpose(a, [..a.Shape[..^2], a.LastDimension, a.PrevDimension]);

    public static CuTensor Transpose(CuTensor a, int[] axis)
    {
        if (axis.Length != a.Dimensions)
            throw new InvalidOperationException($"Axis {axis.Serialize()} is not valid argument for {a.Shape.Serialize()} shape tensor");

        if (!axis.AreElementsUnique())
            throw new InvalidOperationException($"Axis {axis.Serialize()} does not contain all axes for {a.Shape.Serialize()} shape tensor");

        var shape = new int[a.Dimensions];
        for (var i = 0; i < a.Dimensions; ++i)
        {
            shape[i] = a.Shape[axis[i]];
        }

        return CreateNode(
            shape,
            children: [a],
            forward: self => CuBackend.Transpose(self.A, axis, self),
            backward: Ops.NotSupported);
    }

    public static CuTensor Sum(CuTensor a) => Ops.Sum(a);

    public static CuTensor Sum(CuTensor a, int[] axis) => Ops.Sum(a, axis);

    public static CuTensor MatMul(CuTensor a, CuTensor b)
    {
        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);

        return CreateNode(
            [..batchDimensions, a.PrevDimension, b.LastDimension],
            children: [a, b],
            forward: static self => CuBackend.ExecuteMatMul(self.A, self.B, self),
            backward: static (grad, self) => 
            [
                MatMul(grad, Transpose(self.B)),
                MatMul(Transpose(self.A), grad)
            ]);
    }

    public void ApplyOffset(CuTensor offset) => CuBackend.ExecuteAdd(this, offset, this);

    public void ApplyScale(CuTensor scale) => CuBackend.ExecuteMultiply(this, scale, this);
}
