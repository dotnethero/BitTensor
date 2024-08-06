using BitTensor.CUDA.ComputeOnly.Wrappers;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorElementwiseAdd : IDisposable
{
    internal readonly CuTensorDescriptor LeftDescriptor;
    internal readonly CuTensorDescriptor RightDescriptor;
    internal readonly CuTensorDescriptor ResultDescriptor;

    internal readonly CuTensorTernaryOperation Operation;
    internal readonly CuTensorPlan OperationPlan;
    
    public CuTensorElementwiseAdd(CuTensorContext context, CuTensor left, CuTensor right, CuTensor result)
    {
        LeftDescriptor = context.CreateDescriptor(left);
        RightDescriptor = context.CreateDescriptor(right);
        ResultDescriptor = context.CreateDescriptor(result);

        Operation = context.CreateElementwiseAdd(LeftDescriptor, RightDescriptor, ResultDescriptor, ResultDescriptor);
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