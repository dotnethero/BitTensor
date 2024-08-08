using BitTensor.CUDA.Plans;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

internal static class CuBackend
{
    public static void AddInplace(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddInplacePlan(context, a, z);
        plan.Execute(a, z);
    }

    public static void Add(CuTensor a, CuTensor b, CuTensor z, float beta = 1f)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddPlan(context, a, b, z);
        plan.Execute(a, b, z, beta);
    }

    public static void Multiply(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorMatMulPlan(context, a, b, z);
        plan.Execute(a, b, z);
    }
    
    public static void Outer(CuTensor a, CuTensor b, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorOuterProduct(context, a, b, z);
        plan.Execute(a, b, z);
    }

    public static void Sum(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, []);
        plan.Execute(a, z);
    }

    public static void Sum(CuTensor a, HashSet<int> axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorSumPlan(context, a, z, axis);
        plan.Execute(a, z);
    }

    public static void Broadcast(CuTensor a, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorAddInplacePlan(context, a, z);
        plan.Execute(a, z, gamma: 0);
    }

    public static void Transpose(CuTensor a, int[] axis, CuTensor z)
    {
        using var context = new CuTensorContext();
        using var plan = new CuTensorPermutationPlan(context, a, z, axis);
        plan.Execute(a, z);
    }
}