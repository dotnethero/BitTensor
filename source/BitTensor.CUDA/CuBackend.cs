﻿using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

public static class CuBackend
{
    public static void AddInplace(CuTensor a, CuTensor z, float scale = 1f)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorOffsetPlan(context, a, z);
        plan.Execute(a, z, alpha: scale);
    }

    public static void ElementwiseSum(CuTensor a, CuTensor b, CuTensor z, float beta = 1f)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, z);
        plan.Execute(a, b, z, beta);
    }
    
    public static void ElementwiseProduct(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorMultiplyPlan(context, a, b, z);
        plan.Execute(a, b, z);
    }

    public static void MatrixProduct(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorMatrixProductPlan(context, a, b, z);
        plan.Execute(a, b, z);
    }
    
    public static void DotProduct(CuTensor a, CuTensor b, CuTensor z, float scale = 1f)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorContractionPlan(context, a, b, z);
        plan.Execute(a, b, z, alpha: scale);
    }

    public static void OuterProduct(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorOuterProductPlan(context, a, b, z);
        plan.Execute(a, b, z);
    }

    public static void Sum(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, []);
        plan.Execute(a, z);
    }

    public static void Sum(CuTensor a, HashSet<int> axis, CuTensor z, float scale = 1f)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, axis);
        plan.Execute(a, z, scale);
    }
    
    public static void Product(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorProductPlan(context, a, z, []);
        plan.Execute(a, z);
    }

    public static void Product(CuTensor a, HashSet<int> axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorProductPlan(context, a, z, axis);
        plan.Execute(a, z);
    }
    
    public static void Sigmoid(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorUnaryPlusPlan(context, a, z, cutensorOperator_t.CUTENSOR_OP_SIGMOID);
        plan.Execute(a, z, gamma: 0); // replace: z = Sigmoid(a)
    }

    public static void Broadcast(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorOffsetPlan(context, a, z);
        plan.Execute(a, z, gamma: 0);
    }

    public static void Transpose(CuTensor a, int[] axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorPermutationPlan(context, a, z, axis);
        plan.Execute(a, z);
    }
}