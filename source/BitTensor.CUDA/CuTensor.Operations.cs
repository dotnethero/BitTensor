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
        Add(a, b, output);
        return output;
    }

    public static CuTensor operator -(CuTensor a, CuTensor b)
    {
        var shape = Shapes.Broadcast(a.Shape, b.Shape);
        var output = new CuTensor(shape);
        Subtract(a, b, output);
        return output;
    }

    public static CuTensor operator *(CuTensor a, CuTensor b)
    {
        var outputShape = GetMultiplicationShape(a, b);
        var output = new CuTensor(outputShape);
        Multiply(a, b, output);
        return output;
    }

    internal static Shape GetMultiplicationShape(AbstractTensor a, AbstractTensor b)
    {
        if (a.IsScalar)
            return b.Shape;

        if (b.IsScalar)
            return a.Shape;
        
        if (a.IsVector)
            return b.Shape[1..];

        if (b.IsVector) 
            return a.Shape[..^1];
        
        if (a.LastDimension != b.PrevDimension)
            throw new InvalidOperationException($"Shapes are not compatible: {a.Shape} and {b.Shape}");

        var batches = Shapes.Broadcast(a.Shape[..^2], b.Shape[..^2]);

        return [..batches, a.Shape[^2], b.Shape[^1]];
    }

    public static CuTensor Sum(CuTensor a)
    {
        var output = new CuTensor([]);
        Sum(a, output);
        return output;
    }

    public static CuTensor Sum(CuTensor a, HashSet<int> axis)
    {
        var shape = a.Shape.Reduce(axis);
        var output = new CuTensor(shape);
        Sum(a, axis, output);
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
        Transpose(a, axis, output);
        return output;
    }

    public static CuTensor Reshape(CuTensor a, Shape shape)
    {
        if (shape.ArraySize != a.Size)
            throw new InvalidOperationException($"Shape {shape} does not produce {a.Size} size");

        return new CuTensor(shape, a.Pointer);
    }

    // inplace operations
    
    internal static void AddInplace(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddInplacePlan(context, a, z);
        plan.Execute(a, z);
    }

    internal static void Add(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, z);
        plan.Execute(a, b, z, beta: +1);
    }
    
    internal static void Subtract(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, z);
        plan.Execute(a, b, z, beta: -1);
    }
    
    internal static void Multiply(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorMatMulPlan(context, a, b, z);
        plan.Execute(a, b, z);
    }

    internal static void Sum(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, []);
        plan.Execute(a, z);
    }

    internal static void Sum(CuTensor a, HashSet<int> axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, axis);
        plan.Execute(a, z);
    }
    
    internal static void Transpose(CuTensor a, int[] axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorPermutationPlan(context, a, z, axis);
        plan.Execute(a, z);
    }
}