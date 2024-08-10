using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal unsafe class CuTensorTernaryOperation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorTernaryOperation(
        CuTensorContext context,
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        CuTensorDescriptor d,
        cutensorOperator_t opAB,
        cutensorOperator_t opABC)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseTrinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            d.Descriptor, d.Modes, opAB, opABC,
            CUTENSOR_COMPUTE_DESC_32F);

        CuTensorStatus.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public void Execute(CuTensorPlan plan, CuTensor a, CuTensor b, CuTensor c, CuTensor d, float alpha = 1f, float beta = 1f, float gamma = 0f)
    {
        var status = cutensorElementwiseTrinaryExecute(
            Context.Handle,
            plan.Plan, 
            &alpha, a.Pointer, 
            &beta,  b.Pointer, 
            &gamma, c.Pointer, 
            d.Pointer, 
            CuStream.Default);

        CuTensorStatus.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}