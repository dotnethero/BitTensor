using System.Runtime.CompilerServices;
using BitTensor.Abstractions;

namespace BitTensor.CUDA;

public unsafe partial class CuTensor
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static CuTensor operator +(CuTensor a, CuTensor b) => ElementwiseSum(a, b, beta: +1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static CuTensor operator -(CuTensor a, CuTensor b) => ElementwiseSum(a, b, beta: -1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        if (a.IsScalar ||
            b.IsScalar)
            return ElementwiseProduct(a, b);

        if (a.IsVector && 
            b.IsVector)
            return MatrixProduct(a, b);

        return MatrixProduct(a, b);
    }

    public static CuTensor ElementwiseSum(CuTensor a, CuTensor b, float beta = 1f)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.ElementwiseSum(a, b, output, beta);
        return output;
    }
    
    public static CuTensor ElementwiseProduct(CuTensor a, CuTensor b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.ElementwiseProduct(a, b, output);
        return output;
    }

    public static CuTensor MatrixProduct(CuTensor a, CuTensor b)
    {
        var shape = Shapes.BroadcastMatrixProduct(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.MatrixProduct(a, b, output);
        return output;
    }

    public static CuTensor OuterProduct(CuTensor a, CuTensor b)
    {
        var shape = Shapes.BroadcastOuterProduct(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.OuterProduct(a, b, output);
        return output;
    }

    public static CuTensor Sum(CuTensor a)
    {
        var output = new CuTensor([]);
        CuBackend.Sum(a, output);
        return output;
    }

    public static CuTensor Sum(CuTensor a, HashSet<int> axis)
    {
        var shape = a.Shape.Reduce(axis);
        var output = new CuTensor(shape);
        CuBackend.Sum(a, axis, output);
        return output;
    }
    
    public static CuTensor Product(CuTensor a)
    {
        var output = new CuTensor([]);
        CuBackend.Product(a, output);
        return output;
    }

    public static CuTensor Product(CuTensor a, HashSet<int> axis)
    {
        var shape = a.Shape.Reduce(axis);
        var output = new CuTensor(shape);
        CuBackend.Product(a, axis, output);
        return output;
    }

    public static CuTensor Broadcast(CuTensor a, Shape shape)
    {
        if (!a.Shape.CanBroadcastTo(shape))
            throw new InvalidOperationException($"Can't broadcast {a.Shape} to {shape}");

        var output = new CuTensor(shape);
        CuBackend.Broadcast(a, output);
        return output;
    }

    public static CuTensor Transpose(CuTensor a)
    {
        var axis = a.Shape.GetTransposeAxis();
        var shape = a.Shape.Transpose(axis);
        var output = new CuTensor(shape);
        CuBackend.Transpose(a, axis, output);
        return output;
    }

    public static CuTensor Transpose(CuTensor a, int[] axis)
    {
        if (axis.Length != a.Dimensions)
            throw new InvalidOperationException($"Axis {axis.ToText()} is not valid argument for {a.Shape} shape tensor");

        if (!axis.AllElementsAreUnique())
            throw new InvalidOperationException($"Axis {axis.ToText()} does not contain all axes for {a.Shape} shape tensor");

        var shape = a.Shape.Transpose(axis);
        var output = new CuTensor(shape);
        CuBackend.Transpose(a, axis, output);
        return output;
    }
}