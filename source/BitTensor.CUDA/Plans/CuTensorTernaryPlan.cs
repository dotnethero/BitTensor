using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorTernaryPlan<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    internal readonly CuTensorDescriptor<T> LeftDescriptor;
    internal readonly CuTensorDescriptor<T> RightDescriptor;
    internal readonly CuTensorDescriptor<T> ResultDescriptor;
    internal readonly CuTensorTernaryOperation<T> Operation;
    internal readonly CuTensorPlan OperationPlan;

    internal CuTensorTernaryPlan(
        CuTensorContext context,
        AbstractTensor left,
        AbstractTensor right,
        AbstractTensor result,
        cutensorOperator_t opAB,
        cutensorOperator_t opABC)
    {
        LeftDescriptor = new(context, left);
        RightDescriptor = new(context, right);
        ResultDescriptor = new(context, result);

        Operation = new(
            context,
            LeftDescriptor,
            RightDescriptor,
            ResultDescriptor,
            ResultDescriptor,
            opAB,
            opABC);

        OperationPlan = Operation.CreatePlan();
    }
    
    public void Execute(
        IDeviceArray<T> left,
        IDeviceArray<T> right,
        IDeviceArray<T> result,
        float alpha = 1f,
        float beta = 1f,
        float gamma = 0f) =>
        Operation.Execute(
            OperationPlan,
            left,
            right,
            result,
            result,
            alpha,
            beta,
            gamma);

    public void Dispose()
    {
        OperationPlan.Dispose();
        Operation.Dispose();
        ResultDescriptor.Dispose();
        RightDescriptor.Dispose();
        LeftDescriptor.Dispose();
    }
}