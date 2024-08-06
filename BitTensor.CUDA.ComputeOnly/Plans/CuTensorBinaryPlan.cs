using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorAddInplacePlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right) : 
    CuTensorBinaryPlan(context, left, right, cutensorOperator_t.CUTENSOR_OP_ADD);

internal sealed class CuTensorMultiplyInplacePlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right) : 
    CuTensorBinaryPlan(context, left, right, cutensorOperator_t.CUTENSOR_OP_MUL);

internal abstract class CuTensorBinaryPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;

    internal readonly CuTensorBinaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    protected CuTensorBinaryPlan(CuTensorContext context, CuTensor left, CuTensor right, cutensorOperator_t op)
    {
        LeftDescriptor = context.CreateDescriptor(left);
        RightDescriptor = context.CreateDescriptor(right);

        Operation = new CuTensorBinaryOperation(
            context,
            LeftDescriptor,
            RightDescriptor,
            RightDescriptor,
            op);

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