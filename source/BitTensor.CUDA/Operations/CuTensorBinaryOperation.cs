using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal sealed unsafe class CuTensorBinaryOperation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorBinaryOperation(
        CuTensorContext context,
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        cutensorOperator_t opA,
        cutensorOperator_t opB,
        cutensorOperator_t opAB)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseBinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, opA,
            b.Descriptor, b.Modes, opB,
            c.Descriptor, c.Modes, opAB,
            CUTENSOR_COMPUTE_DESC_32F);

        Status.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public void Execute(CuTensorPlan plan, CuTensor a, CuTensor b, CuTensor c, float alpha = 1f, float gamma = 1f)
    {
        var status = cutensorElementwiseBinaryExecute(Context.Handle, plan.Plan, &alpha, a.Pointer, &gamma, b.Pointer, c.Pointer, CuStream.Default);
        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}