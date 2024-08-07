using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

internal sealed class CuTensorMatMulPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorContraction Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;

    public CuTensorMatMulPlan(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        var (leftModes, rightModes, resultModes) = GetModes(left, right, result);

        LeftDescriptor = context.CreateDescriptor(left, leftModes);
        RightDescriptor = context.CreateDescriptor(right, rightModes);
        ResultDescriptor = context.CreateDescriptor(result, resultModes);

        Contraction = context.CreateContraction(LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    private static (int[] leftModes, int[] rightModes, int[] resultModes) GetModes(AbstractTensor left, AbstractTensor right, AbstractTensor result)
    {
        var leftModes = left.Shape.GetOrdinaryModes();
        var rightModes = right.Shape.GetOrdinaryModes();
        var resultModes = result.Shape.GetOrdinaryModes();

        var contractionMode = Math.Max(left.Dimensions, right.Dimensions) + 1;

        if (left.Dimensions > 0)
        {
            if (right.Dimensions == 1)
            {
                leftModes[^1] = contractionMode;
                rightModes[^1] = contractionMode;
                resultModes = result.Shape.GetOrdinaryModes(offset: +1);
            }
            if (right.Dimensions > 1)
            {
                leftModes[^1] = contractionMode;
                rightModes[^2] = contractionMode;
                resultModes = result.Shape.GetOrdinaryModes();
            }
        }

        return (leftModes, rightModes, resultModes);
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
