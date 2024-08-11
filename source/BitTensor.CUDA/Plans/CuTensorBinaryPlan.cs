using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

public sealed class CuTensorBinaryPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;

    internal readonly CuTensorBinaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    internal CuTensorBinaryPlan(CuTensorContext context,
        AbstractTensor a,
        AbstractTensor b,
        Ops opA,
        Ops opB,
        Ops opAB)
    {
        LeftDescriptor = context.CreateDescriptor(a);
        RightDescriptor = context.CreateDescriptor(b);

        Operation = new CuTensorBinaryOperation(
            context,
            LeftDescriptor,
            RightDescriptor,
            RightDescriptor,
            opA,
            opB,
            opAB);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(CuTensor left, CuTensor right, float alpha = 1f, float gamma = 1f) =>
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