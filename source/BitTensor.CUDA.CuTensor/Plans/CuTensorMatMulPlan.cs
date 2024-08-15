using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorMatMulPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> A;
    internal readonly CuTensorDescriptor<T> B;
    internal readonly CuTensorDescriptor<T> C;
    internal readonly CuTensorContraction<T> Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;
    internal bool IsDisposed;

    internal CuTensorMatMulPlan(CuTensorContext context, Operand a, Operand b, Shape c)
    {
        if (a.Shape.Dimensions < 2 ||
            b.Shape.Dimensions < 2)
            throw new InvalidOperationException("Can't execute matrix multiplication on vectors and scalars - use dimension padding");

        var leftModes = a.Shape.GetOrdinaryModes();
        var rightModes = b.Shape.GetOrdinaryModes();
        var resultModes = c.GetOrdinaryModes();

        // contraction
        leftModes[^1] = -1;
        rightModes[^2] = -1;

        A = new(context, a, leftModes);
        B = new(context, b, rightModes);
        C = new(context, c, resultModes);
        Contraction = new(context, A, B, C, C);
        ContractionPlan = Contraction.CreatePlan();
        Workspace = Contraction.CreateWorkspace(ContractionPlan);
    }

    public void Execute(
        IDeviceArray<T> a,
        IDeviceArray<T> b,
        IDeviceArray<T> c,
        float alpha = 1f,
        float beta = 0f) =>
        Contraction.Execute(ContractionPlan, Workspace, a, b, c, c, alpha, beta);

    public void Dispose()
    {
        if (IsDisposed) return;

        ContractionPlan.Dispose();
        Workspace.Dispose();
        Contraction.Dispose();
        A.Dispose();
        B.Dispose();
        C.Dispose();
        IsDisposed = true;
    }
}
