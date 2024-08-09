using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using static cuTENSOR;

public unsafe class CuTensorContraction : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorContraction(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, CuTensorDescriptor d)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateContraction(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            c.Descriptor, c.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY, 
            d.Descriptor, d.Modes, CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public void Execute(CuTensor a, CuTensor b, CuTensor c, CuTensor d, float alpha = 1f, float beta = 1f)
    {
        using var plan = CreatePlan();
        using var ws = CreateWorkspace(plan);

        ExecuteByPlan(plan, ws, a, b, c, d, alpha, beta);
    }

    public void ExecuteByPlan(CuTensorPlan plan, CuTensorWorkspace ws, CuTensor a, CuTensor b, CuTensor c, CuTensor d, float alpha = 1f, float beta = 1f)
    {
        var status = cutensorContract(
            Context.Handle, 
            plan.Plan, 
            &alpha, a.Pointer, b.Pointer, 
            &beta,  c.Pointer, d.Pointer, 
            ws.Pointer, 
            ws.Bytes,
            CuStream.Default);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    internal CuTensorPlan CreatePlan() => new(this);

    internal CuTensorWorkspace CreateWorkspace(CuTensorPlan plan) => new(plan.WorkspaceSize);

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}