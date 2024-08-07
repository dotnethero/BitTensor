using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorBinaryOperation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorBinaryOperation(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, cutensorOperator_t operation)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseBinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, operation,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public void Execute(CuTensor a, CuTensor b, CuTensor c, float alpha = 1f, float gamma = 1f)
    {
        using var plan = CreatePlan();

        ExecuteByPlan(plan, a, b, c, alpha, gamma);
    }

    public void ExecuteByPlan(CuTensorPlan plan, CuTensor a, CuTensor b, CuTensor c, float alpha = 1f, float gamma = 1f)
    {
        var status = cutensorElementwiseBinaryExecute(Context.Handle, plan.Plan, &alpha, a.Pointer, &gamma, b.Pointer, c.Pointer, CuStream.Default);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    internal CuTensorPlan CreatePlan() => new(this);

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}