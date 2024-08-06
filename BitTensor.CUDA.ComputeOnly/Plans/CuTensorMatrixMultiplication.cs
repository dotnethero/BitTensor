using BitTensor.Abstractions;
using BitTensor.CUDA.ComputeOnly.Wrappers;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorMatrixMultiplication : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    public CuTensorMatrixMultiplication(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        LeftDescriptor = PrepareLeft(context, left);
        RightDescriptor = PrepareRight(context, right);
        ResultDescriptor = PrepareResult(context, result);

        Contraction = context.CreateContraction(LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    public void Execute(CuTensor left, CuTensor right, CuTensor result)
    {
        Contraction.ExecuteWithPlan(
            ContractionPlan,
            Workspace,
            left,
            right,
            result,
            result,
            alpha: 1,
            beta: 0);
    }

    private static CuTensorDescriptor PrepareLeft(CuTensorContext context, CuTensor left)
    {
        var modes = left.Shape.GetModes(offset: 1);
        modes[^2] = 1;
        modes[^1] = 2;

        return context.CreateDescriptor(left, modes);
    }

    private static CuTensorDescriptor PrepareRight(CuTensorContext context, CuTensor right)
    {
        var modes = right.Shape.GetModes(offset: 1);
        modes[^2] = 2;
        modes[^1] = 3;

        return context.CreateDescriptor(right, modes);
    }

    private static CuTensorDescriptor PrepareResult(CuTensorContext context, CuTensor result)
    {
        var modes = result.Shape.GetModes(offset: 1);
        modes[^2] = 1;
        modes[^1] = 3;

        return context.CreateDescriptor(result, modes);
    }

    public void Dispose()
    {
        ContractionPlan.Dispose();
        Workspace.Dispose();
        Contraction.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}
