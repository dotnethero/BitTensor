using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorMatMulPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> LeftDescriptor;
    internal readonly CuTensorDescriptor<T> RightDescriptor;
    internal readonly CuTensorDescriptor<T> ResultDescriptor;
    internal readonly CuTensorContraction<T> Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;
    internal bool IsDisposed;

    internal CuTensorMatMulPlan(CuTensorContext context, Shape left, Shape right, Shape result)
    {
        if (left.Dimensions < 2 ||
            right.Dimensions < 2)
            throw new InvalidOperationException("Can't execute matrix multiplication on vectors and scalars - use dimension padding");

        var leftModes = left.GetOrdinaryModes();
        var rightModes = right.GetOrdinaryModes();
        var resultModes = result.GetOrdinaryModes();

        // contraction
        leftModes[^1] = -1;
        rightModes[^2] = -1;

        LeftDescriptor = new(context, left, leftModes);
        RightDescriptor = new(context, right, rightModes);
        ResultDescriptor = new(context, result, resultModes);

        Contraction = new(context, LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    public void Execute(
        IDeviceArray<T> left,
        IDeviceArray<T> right,
        IDeviceArray<T> result,
        float alpha = 1f,
        float beta = 0f) =>
        Contraction.Execute(
            ContractionPlan,
            Workspace,
            left,
            right,
            result,
            result,
            alpha,
            beta);

    public void Dispose()
    {
        if (IsDisposed) return;

        ContractionPlan.Dispose();
        Workspace.Dispose();
        Contraction.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
        IsDisposed = true;
    }
}
