using BitTensor.Abstractions;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public unsafe partial class CuTensor
{
    public static CuTensor operator +(CuTensor a, CuTensor b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.Add(a, b, output);
        return output;
    }

    public static CuTensor operator -(CuTensor a, CuTensor b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        CuBackend.Subtract(a, b, output);
        return output;
    }

    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        var outputShape = Shapes.BroadcastMatMul(a.Shape, b.Shape);
        var output = new CuTensor(outputShape);
        CuBackend.Multiply(a, b, output);
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

    public static CuTensor Reshape(CuTensor a, Shape shape)
    {
        if (shape.ArraySize != a.Size)
            throw new InvalidOperationException($"Shape {shape} does not produce {a.Size} size");

        return new CuTensor(shape, a.Pointer);
    }
}