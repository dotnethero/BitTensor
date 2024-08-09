using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

using Ops = cutensorOperator_t;

internal sealed class CuTensorUnaryPlusPlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right,
    Ops operation) : 
    CuTensorBinaryPlan(context, left, right, operation, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

internal sealed class CuTensorOffsetPlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right) : 
    CuTensorBinaryPlan(context, left, right, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

internal sealed class CuTensorScalePlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right) : 
    CuTensorBinaryPlan(context, left, right, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_MUL);

internal abstract class CuTensorBinaryPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;

    internal readonly CuTensorBinaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    protected CuTensorBinaryPlan(CuTensorContext context,
        CuTensor a,
        CuTensor b,
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
        Operation.ExecuteByPlan(
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