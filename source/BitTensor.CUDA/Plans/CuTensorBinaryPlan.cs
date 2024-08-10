using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

public sealed class CuTensorUnaryPlusPlan(
    CuTensorContext context,
    AbstractTensor left,
    AbstractTensor right,
    Ops operation) : 
    CuTensorBinaryPlan(context, left, right, operation, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

public sealed class CuTensorOffsetPlan(
    CuTensorContext context,
    AbstractTensor left,
    AbstractTensor right) : 
    CuTensorBinaryPlan(context, left, right, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

public sealed class CuTensorScalePlan(
    CuTensorContext context,
    AbstractTensor left,
    AbstractTensor right) : 
    CuTensorBinaryPlan(context, left, right, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_MUL);

public abstract class CuTensorBinaryPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;

    internal readonly CuTensorBinaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    protected CuTensorBinaryPlan(CuTensorContext context,
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