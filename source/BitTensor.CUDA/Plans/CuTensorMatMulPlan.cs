using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorMatMulPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;
    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    internal CuTensorMatMulPlan(CuTensorContext context, AbstractTensor left, AbstractTensor right, AbstractTensor result)
    {
        if (left.Shape.Dimensions < 2 ||
            right.Shape.Dimensions < 2)
            throw new InvalidOperationException("Can't execute matrix multiplication on vectors and scalars - use dimension padding");

        var leftModes = left.Shape.GetOrdinaryModes();
        var rightModes = right.Shape.GetOrdinaryModes();
        var resultModes = result.Shape.GetOrdinaryModes();

        // contraction
        leftModes[^1] = -1;
        rightModes[^2] = -1;

        LeftDescriptor = context.CreateDescriptor(left, leftModes);
        RightDescriptor = context.CreateDescriptor(right, rightModes);
        ResultDescriptor = context.CreateDescriptor(result, resultModes);

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
