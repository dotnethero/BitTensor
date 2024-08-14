using System.Numerics;
using BitTensor.Abstractions;

namespace BitTensor.CUDA.Graph;

public static partial class Ops
{
    public static CudaNode<T> Transpose<T>(CudaNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var axis = a.Shape.GetTransposeAxis();
        return Transpose(a, axis);
    }
    
    public static CudaNode<T> Transpose<T>(CudaNode<T> a, Index[] axis) where T : unmanaged, IFloatingPoint<T>
    {
        if (axis.Length != a.Dimensions)
            throw new InvalidOperationException($"Axis {axis.ToText()} is not valid argument for {a.Shape} shape tensor");

        if (!a.Shape.AxisAreUnique(axis))
            throw new InvalidOperationException($"Axis {axis.ToText()} does not contain all axes for {a.Shape} shape tensor");

        var context = a.Context;
        var shape = a.Shape.Transpose(axis);
        var inverted = Axis.InvertPermutation(axis);
        var plan = context.cuTENSOR.CreatePermutationPlan<T>(a.Shape, shape, axis);
        return new(
            context,
            shape,
            children: [a],
            forward: (output) => plan.Execute(a, output),
            backward: (grad, _) => [Transpose(grad, inverted)]);
    }

}