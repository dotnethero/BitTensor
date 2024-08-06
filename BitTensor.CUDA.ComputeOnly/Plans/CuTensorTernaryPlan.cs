using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorAddPlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right,
    CuTensor result) : 
    CuTensorTernaryPlan(context, left, right, result, cutensorOperator_t.CUTENSOR_OP_ADD);

internal sealed class CuTensorMultiplyPlan(
    CuTensorContext context,
    CuTensor left,
    CuTensor right,
    CuTensor result) : 
    CuTensorTernaryPlan(context, left, right, result, cutensorOperator_t.CUTENSOR_OP_MUL);

internal abstract class CuTensorTernaryPlan : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorTernaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;

    protected CuTensorTernaryPlan(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result, cutensorOperator_t op)
    {
        LeftDescriptor = context.CreateDescriptor(left);
        RightDescriptor = context.CreateDescriptor(right);
        ResultDescriptor = context.CreateDescriptor(result);

        Operation = new CuTensorTernaryOperation(
            context,
            LeftDescriptor,
            RightDescriptor,
            ResultDescriptor,
            ResultDescriptor,
            op,
            cutensorOperator_t.CUTENSOR_OP_ADD);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(CuTensor left, CuTensor right, CuTensor result)
    {
        Operation.ExecuteWithPlan(
            OperationPlan,
            left,
            right,
            result,
            result,
            alpha: 1,
            beta: 1,
            gamma: 0);
    }
    
    public void Dispose()
    {
        OperationPlan.Dispose();
        Operation.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}