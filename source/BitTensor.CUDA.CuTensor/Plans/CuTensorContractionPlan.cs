using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorContractionPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> A;
    internal readonly CuTensorDescriptor<T> B;
    internal readonly CuTensorDescriptor<T> C;
    internal readonly CuTensorContraction<T> Contraction;
    internal readonly CuTensorPlan ContractionPlan;
    internal readonly CuTensorWorkspace Workspace;
    internal bool IsDisposed;

    internal CuTensorContractionPlan(CuTensorContext context, Operand a, Operand b, Operand c)
    {
        A = new(context, a);
        B = new(context, b);
        C = new(context, c);
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