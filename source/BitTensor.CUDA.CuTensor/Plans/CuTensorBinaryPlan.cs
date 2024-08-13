using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

public sealed class CuTensorBinaryPlan<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    internal readonly CuTensorDescriptor<T> LeftDescriptor;
    internal readonly CuTensorDescriptor<T> RightDescriptor;
    internal readonly CuTensorBinaryOperation<T> Operation;
    internal readonly CuTensorPlan OperationPlan;

    internal CuTensorBinaryPlan(
        CuTensorContext context,
        AbstractTensor a,
        AbstractTensor b,
        Ops opA,
        Ops opB,
        Ops opAB)
    {
        LeftDescriptor = new(context, a);
        RightDescriptor = new(context, b);
        Operation = new(
            context,
            LeftDescriptor,
            RightDescriptor,
            RightDescriptor,
            opA,
            opB,
            opAB);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(IDeviceArray<T> left, IDeviceArray<T> right, float alpha = 1f, float gamma = 1f) =>
        Operation.Execute(
            OperationPlan,
            left,
            right,
            right,
            alpha,
            gamma);

    public void Dispose()
    {
        OperationPlan.Dispose();
        Operation.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}