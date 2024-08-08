using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

internal sealed class CuTensorOuterProduct : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    public CuTensorOuterProduct(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        var leftModes = left.Shape.GetOrdinaryModes();
        var rightModes = right.Shape.GetOrdinaryModes(offset: left.Dimensions);

        LeftDescriptor = context.CreateDescriptor(left, leftModes);
        RightDescriptor = context.CreateDescriptor(right, rightModes);
        ResultDescriptor = context.CreateDescriptor(result, [..leftModes, ..rightModes]);

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
