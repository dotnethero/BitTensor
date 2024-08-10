using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal unsafe class CuTensorPermutation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorPermutation(
        CuTensorContext context,
        CuTensorDescriptor a,
        CuTensorDescriptor b)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreatePermutation(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes,
            CUTENSOR_COMPUTE_DESC_32F);

        Status.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public void Execute(CuTensorPlan plan, CuTensor a, CuTensor b, float alpha = 1f)
    {
        var status = cutensorPermute(Context.Handle, plan.Plan, &alpha, a.Pointer, b.Pointer, CuStream.Default);
        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}