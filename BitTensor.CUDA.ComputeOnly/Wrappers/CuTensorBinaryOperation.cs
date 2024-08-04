using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorBinaryOperation : IDisposable
{
    internal readonly CuTensorContext Context;
    internal readonly cutensorOperationDescriptor* Descriptor;

    public CuTensorBinaryOperation(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, cutensorOperator_t operation)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseBinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, operation,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public CuTensorPlan CreatePlan() => new(this);

    public void Execute(CuTensor a, CuTensor b, CuTensor c)
    {
        using var plan = CreatePlan();
        plan.Execute(a, b, c);
    }

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}