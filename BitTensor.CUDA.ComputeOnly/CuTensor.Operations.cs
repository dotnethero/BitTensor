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
    
    public static CuTensor operator *(CuTensor a, float b)
    {
        var output = new CuTensor(a.Shape);
        Scale(a, b, output);
        return output;
    }

    public static CuTensor Sum(CuTensor a, int[] axis) => Sum(a, new HashSet<int>(axis));

    public static CuTensor Sum(CuTensor a, HashSet<int> axis)
    {
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

    public static void Add(CuTensor a, CuTensor b, CuTensor c)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var c1 = context.CreateDescriptor(c);

        using var operation = context.CreateElementwiseAdd(a1, b1, c1, c1);

        operation.Execute(a, b, c, c, gamma: 0);
    }
    
    public static void Subtract(CuTensor a, CuTensor b, CuTensor c)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var c1 = context.CreateDescriptor(c);

        using var operation = context.CreateElementwiseAdd(a1, b1, c1, c1);

        operation.Execute(a, b, c, c, gamma: 0, beta: -1);
    }

    public static void Contract(CuTensor a, CuTensor b, CuTensor c, CuTensor d)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var b1 = context.CreateDescriptor(b);
        using var c1 = context.CreateDescriptor(c);
        using var d1 = context.CreateDescriptor(d);

        using var operation = context.CreateContraction(a1, b1, c1, d1);

        operation.Execute(a, b, c, d);
    }
    
    private static void Sum(CuTensor a, HashSet<int> axis, CuTensor output)
    {
        throw new NotImplementedException();
    }

    public static void Scale(CuTensor a, float b, CuTensor c)
    {
        var context = new CublasContext();

        context.Axpy(a, b, c);
    }

    public static void Multiply(CuTensor a, CuTensor b, CuTensor c)
    {
        var context = new CublasContext();

        context.Gemm(a, b, c);
    }
}