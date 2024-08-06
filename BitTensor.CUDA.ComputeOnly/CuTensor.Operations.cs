using BitTensor.Abstractions;
using BitTensor.CUDA.ComputeOnly.Wrappers;

namespace BitTensor.CUDA.ComputeOnly;

public unsafe partial class CuTensor
{
    public static CuTensor operator +(CuTensor a, CuTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        Add(a, b, output);
        return output;
    }

    public static CuTensor operator -(CuTensor a, CuTensor b)
    {
        var shape = Shapes.EnsureShapesAreCompatible(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        Subtract(a, b, output);
        return output;
    }

    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        var batchDimensions = Shapes.EnsureShapesAreCompatible(a.Shape[..^2], b.Shape[..^2]);
        var output = new CuTensor([..batchDimensions, a.PrevDimension, b.LastDimension]);
        Multiply(a, b, output);
        return output;
    }
    
    public static CuTensor Sum(CuTensor a, int[] axis) => Sum(a, new HashSet<int>(axis));
    
    public static CuTensor Sum(CuTensor a)
    {
        if (a.IsScalar)
            return a;

        var output = new CuTensor([]);
        Sum(a, output);
        return output;
    }

    private static CuTensor Sum(CuTensor a, HashSet<int> axis)
    {
        if (axis.Count == 0)
            return a;

        if (axis.Count == a.Dimensions)
            return Sum(a);

        var shape = a.Shape.Where((s, i) => !axis.Contains(i)).ToArray();
        var output = new CuTensor(shape);
        Sum(a, axis, output);
        return output;
    }

    public static CuTensor Reshape(CuTensor a, int[] shape)
    {
        if (shape.Product() != a.Size)
            throw new InvalidOperationException($"Shape {shape.Serialize()} does not produce {a.Size} size");

        return new CuTensor(shape, a.Pointer);
    }

    // inplace operations
    
    public static void Add(CuTensor a, CuTensor r)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var r1 = context.CreateDescriptor(r);

        using var operation = context.CreateElementwiseAdd(a1, r1, r1);

        operation.Execute(a, r, r);
    }

    public static void Add(CuTensor a, CuTensor b, CuTensor r)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var r1 = context.CreateDescriptor(r);

        using var operation = context.CreateElementwiseAdd(a1, b1, r1, r1);

        operation.Execute(a, b, r, r, gamma: 0);
    }
    
    public static void Subtract(CuTensor a, CuTensor b, CuTensor r)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var r1 = context.CreateDescriptor(r);

        using var operation = context.CreateElementwiseAdd(a1, b1, r1, r1);

        operation.Execute(a, b, r, r, gamma: 0, beta: -1);
    }
    
    public static void Multiply(CuTensor a, CuTensor b, CuTensor r)
    {
        var aModes = a.Shape.GetModes(offset: 1);
        aModes[^2] = 1;
        aModes[^1] = 2;

        var bModes = b.Shape.GetModes(offset: 1);
        bModes[^2] = 2;
        bModes[^1] = 3;

        var rModes = r.Shape.GetModes(offset: 1);
        rModes[^2] = 1;
        rModes[^1] = 3;

        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a, aModes);
        using var b1 = context.CreateDescriptor(b, bModes);
        using var r1 = context.CreateDescriptor(r, rModes);

        using var operation = context.CreateContraction(a1, b1, r1, r1);

        operation.Execute(a, b, r, r, beta: 0);
    }
    
    private static void Sum(CuTensor a, CuTensor r)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var r1 = context.CreateDescriptor(r);

        using var operation = context.CreateSum(a1, r1, r1);

        operation.Execute(a, r, r, beta: 0);
    }

    private static void Sum(CuTensor a, HashSet<int> axis, CuTensor r)
    {
        var modes = a.Shape
            .GetModes()
            .Where((s, i) => !axis.Contains(i))
            .ToArray();

        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var r1 = context.CreateDescriptor(r, modes);

        using var operation = context.CreateSum(a1, r1, r1);

        operation.Execute(a, r, r, beta: 0);
    }
}