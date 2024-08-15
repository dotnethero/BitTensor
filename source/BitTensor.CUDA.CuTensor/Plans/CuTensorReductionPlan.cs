using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using OpCode = cutensorOperator_t;

public sealed class CuTensorReductionPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> A;
    internal readonly CuTensorDescriptor<T> B;
    internal readonly CuTensorReduction<T> Reduction;
    internal readonly CuTensorPlan ReductionPlan;
    internal readonly CuTensorWorkspace Workspace;
    internal bool IsDisposed;

    internal CuTensorReductionPlan(CuTensorContext context, Operand a, Operand b, HashSet<Index> axis, OpCode op, bool keepDims = false)
    {
        var modes = a.Shape.GetReductionModes(axis);

        A = new(context, a);
        B = keepDims
            ? new(context, b)
            : new(context, b, modes);

        Reduction = new(context, A, B, B, op);
        ReductionPlan = Reduction.CreatePlan();
        Workspace = Reduction.CreateWorkspace(ReductionPlan);
    }
    
    public void Execute(IDeviceArray<T> input, IDeviceArray<T> output, float alpha = 1f, float beta = 0f) =>
        Reduction.Execute(
            ReductionPlan,
            Workspace,
            input,
            output,
            output,
            alpha,
            beta);

    public void Dispose()
    {
        if (IsDisposed) return;

        ReductionPlan.Dispose();
        Workspace.Dispose();
        Reduction.Dispose();
        A.Dispose();
        B.Dispose();
        IsDisposed = true;
    }
}
