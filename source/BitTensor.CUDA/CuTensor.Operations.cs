using BitTensor.Abstractions;

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
    
    public static CuTensor Outer(CuTensor a, CuTensor b)
    {
        var output = new CuTensor([..a.Shape, ..b.Shape]);
        CuBackend.Outer(a, b, output);
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