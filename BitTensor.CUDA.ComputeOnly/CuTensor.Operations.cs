using BitTensor.Abstractions;
using BitTensor.CUDA.ComputeOnly.Plans;
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

        var output = new CuTensor(shape);
        Transpose(a, axis, output);
        return output;
    }

    public static CuTensor Reshape(CuTensor a, int[] shape)
    {
        if (shape.Product() != a.Size)
            throw new InvalidOperationException($"Shape {shape.Serialize()} does not produce {a.Size} size");

        return new CuTensor(shape, a.Pointer);
    }

    // inplace operations
    
    internal static void Add(CuTensor a, CuTensor r)
    {
        using var context = new CuTensorContext();

        using var a1 = context.CreateDescriptor(a);
        using var r1 = context.CreateDescriptor(r);

        using var operation = context.CreateElementwiseAdd(a1, r1, r1);

        operation.Execute(a, r, r);
    }

    internal static void Add(CuTensor a, CuTensor b, CuTensor r)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, r);

        plan.Execute(a, b, r, beta: +1);
    }
    
    internal static void Subtract(CuTensor a, CuTensor b, CuTensor r)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, r);

        plan.Execute(a, b, r, beta: -1);
    }
    
    internal static void Multiply(CuTensor a, CuTensor b, CuTensor r)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorMatMulPlan(context, a, b, r);

        plan.Execute(a, b, r);
    }

    internal static void Sum(CuTensor a, CuTensor r)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, r, []);

        plan.Execute(a, r);
    }

    internal static void Sum(CuTensor a, HashSet<int> axis, CuTensor r)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, r, axis); // TODO: axis as parameter

        plan.Execute(a, r);
    }
    
    internal static void Transpose(CuTensor input, int[] axis, CuTensor output)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorPermutationPlan(context, input, output, axis);

        plan.Execute(input, output);
    }
}