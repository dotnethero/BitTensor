using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

internal sealed class CuTensorOuterProductPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    public CuTensorOuterProductPlan(CuTensorContext context, AbstractTensor left, AbstractTensor right, AbstractTensor result)
    {
        var leftModes = left.Shape.GetOrdinaryModes();
        var rightModes = right.Shape.GetOrdinaryModes();
        var lastMode = Math.Max(left.Dimensions, right.Dimensions);

        leftModes[^1] = ++lastMode;
        rightModes[^1] = ++lastMode;

        var longest = leftModes.Length > rightModes.Length 
            ? leftModes[..^1] 
            : rightModes[..^1];

        LeftDescriptor = context.CreateDescriptor(left, leftModes);
        RightDescriptor = context.CreateDescriptor(right, rightModes);
        ResultDescriptor = context.CreateDescriptor(result, [..longest, leftModes[^1], rightModes[^1]]);

        Contraction = new(context, LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    public void Execute(CuTensor left, CuTensor right, CuTensor result) =>
        Contraction.Execute(
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
