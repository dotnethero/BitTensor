using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using OpCode = cutensorOperator_t;

public sealed class CuTensorBroadcastPlan<T> : ICuTensorPlan where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> A;
    internal readonly CuTensorDescriptor<T> B;
    internal readonly CuTensorBinaryOperation<T> Operation;
    internal readonly CuTensorPlan OperationPlan;
    internal bool IsDisposed;

    internal CuTensorBroadcastPlan(CuTensorContext context, Operand a, Operand b)
    {
        A = new(context, a);
        B = new(context, b);
        Operation = new(context, A, B, B, OpCode.CUTENSOR_OP_ADD);
        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(IDeviceArray<T> a, IDeviceArray<T> b, float alpha = 1f, float gamma = 0f) =>
        Operation.Execute(OperationPlan, a, b, b, alpha, gamma);

    public void Dispose()
    {
        if (IsDisposed) return;

        OperationPlan.Dispose();
        Operation.Dispose();
        A.Dispose();
        B.Dispose();
        IsDisposed = true;
    }
}