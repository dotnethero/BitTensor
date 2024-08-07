﻿using BitTensor.Abstractions;
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
    
    public static CuTensor Sum(CuTensor a)
    {
        var output = new CuTensor([]);
        Sum(a, output);
        return output;
    }

    public static CuTensor Sum(CuTensor a, HashSet<int> axis)
    {
        var shape = Shapes.Reduce(a.Shape, axis);
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

        var shape = Shapes.Transpose(a.Shape, axis);
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