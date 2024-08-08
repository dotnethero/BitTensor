using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

internal sealed class CuTensorMatrixProductPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    public CuTensorMatrixProductPlan(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        var (leftModes, rightModes, resultModes) = Modes.GetMultiplicationModes(left.Shape, right.Shape, result.Shape);

        LeftDescriptor = context.CreateDescriptor(left, leftModes);
        RightDescriptor = context.CreateDescriptor(right, rightModes);
        ResultDescriptor = context.CreateDescriptor(result, resultModes);

        Contraction = context.CreateContraction(LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    public void Execute(CuTensor left, CuTensor right, CuTensor result) =>
        Contraction.ExecuteByPlan(
            ContractionPlan,
            Workspace,
            left,
            right,
            result,
            result,
            alpha: 1,
            beta: 0);

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
