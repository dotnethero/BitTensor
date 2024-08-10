using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

using static cuTENSOR;

internal unsafe class CuTensorReduction : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorReduction(
        CuTensorContext context,
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        cutensorOperator_t opReduce)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateReduction(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, opReduce,
            CUTENSOR_COMPUTE_DESC_32F);

        Status.EnsureIsSuccess(status);

        Context = context;
        Descriptor = descriptor;
    }

    public CuTensorPlan CreatePlan() => new(this);

    public CuTensorWorkspace CreateWorkspace(CuTensorPlan plan) => new(plan.WorkspaceSize);

    public void Execute(CuTensorPlan plan, CuTensorWorkspace ws, CuTensor a, CuTensor b, CuTensor c, float alpha, float beta)
    {
        var status = cutensorReduce(Context.Handle, plan.Plan, &alpha, a.Pointer, &beta, b.Pointer, c.Pointer, ws.Pointer, ws.Bytes, CuStream.Default);
        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}