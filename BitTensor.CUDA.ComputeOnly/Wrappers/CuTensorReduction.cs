using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorReduction : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorReduction(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, cutensorOperator_t opReduce)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateReduction(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, opReduce,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public void Execute(CuTensor a, CuTensor b, CuTensor c, float alpha = 1f, float beta = 1f)
    {
        using var plan = new CuTensorPlan(this);
        using var ws = new CuTensorWorkspace(plan.WorkspaceSize);

        var status = cutensorReduce(Context.Handle, plan.Plan, &alpha, a.Pointer, &beta, b.Pointer, c.Pointer, ws.Pointer, ws.Bytes, (CUstream_st*) 0);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}