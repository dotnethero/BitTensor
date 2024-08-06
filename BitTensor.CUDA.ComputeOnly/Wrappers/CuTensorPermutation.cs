using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorPermutation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorPermutation(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreatePermutation(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public void Execute(CuTensor a, CuTensor b, float alpha = 1f)
    {
        using var plan = CreatePlan();
        ExecuteByPlan(plan, a, b, alpha);
    }

    public void ExecuteByPlan(CuTensorPlan plan, CuTensor a, CuTensor b, float alpha = 1f)
    {
        var status = cutensorPermute(Context.Handle, plan.Plan, &alpha, a.Pointer, b.Pointer, CuStream.Default);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    public CuTensorPlan CreatePlan() => new(this);

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}