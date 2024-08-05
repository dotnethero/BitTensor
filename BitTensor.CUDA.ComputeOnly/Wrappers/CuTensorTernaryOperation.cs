﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

using static cuTENSOR;

internal unsafe class CuTensorTernaryOperation : ICuTensorOperation
{
    public CuTensorContext Context { get; }
    public cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorTernaryOperation(CuTensorContext context, CuTensorDescriptor a, CuTensorDescriptor b, CuTensorDescriptor c, CuTensorDescriptor d, cutensorOperator_t opAB, cutensorOperator_t opABC)
    {
        cutensorOperationDescriptor* descriptor;

        var status = cutensorCreateElementwiseTrinary(
            context.Handle, 
            &descriptor,
            a.Descriptor, a.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            b.Descriptor, b.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            d.Descriptor, d.Modes, cutensorOperator_t.CUTENSOR_OP_IDENTITY,
            d.Descriptor, d.Modes, opAB, opABC,
            CUTENSOR_COMPUTE_DESC_32F);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Context = context;
        Descriptor = descriptor;
    }
    
    public void Execute(CuTensor a, CuTensor b, CuTensor c, CuTensor d, float alpha = 1f, float beta = 1f, float gamma = 1f)
    {
        using var plan = new CuTensorPlan(this);

        var status = cutensorElementwiseTrinaryExecute(
            Context.Handle,
            plan.Plan, 
            &alpha, a.Pointer, 
            &beta,  b.Pointer, 
            &gamma, c.Pointer, 
            d.Pointer, 
            (CUstream_st*) 0);

        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    public void Dispose()
    {
        cutensorDestroyOperationDescriptor(Descriptor);
    }
}