﻿using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Handle = handle;
    }

    public CuTensorDescriptor CreateDescriptor(CuTensor a) => 
        new(this, a);
    
    public CuTensorDescriptor CreateDescriptor(CuTensor a, int[] modes) => 
        new(this, a, modes);
    
    public CuTensorBinaryOperation CreateElementwiseAdd(
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c) => 
        new(this, a, b, c, cutensorOperator_t.CUTENSOR_OP_ADD);

    public CuTensorTernaryOperation CreateElementwiseAdd(
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        CuTensorDescriptor d) => 
        new(this, a, b, c, d, cutensorOperator_t.CUTENSOR_OP_ADD, cutensorOperator_t.CUTENSOR_OP_ADD);
    
    public CuTensorContraction CreateContraction(
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        CuTensorDescriptor d) => 
        new(this, a, b, c, d);
    
    public CuTensorReduction CreateSum(
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c) => 
        new(this, a, b, c, cutensorOperator_t.CUTENSOR_OP_ADD);

    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}