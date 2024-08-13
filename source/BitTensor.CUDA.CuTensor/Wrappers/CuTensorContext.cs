using BitTensor.Abstractions;
using System.Numerics;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Wrappers;

using Ops = cutensorOperator_t;

public sealed unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        Status.EnsureIsSuccess(status);

        Handle = handle;
    }

    public CuTensorBinaryPlan<T> CreateUnaryPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        Ops unary) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, output, unary, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(this, output, output, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan<T> CreateAddPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, b, output, Ops.CUTENSOR_OP_ADD, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan<T> CreateMultiplyPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, b, output, Ops.CUTENSOR_OP_MUL, Ops.CUTENSOR_OP_ADD);
    
    public CuTensorMatMulPlan<T> CreateMatMulPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, b, output);

    public CuTensorContractionPlan<T> CreateContractionPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output)
        where T : unmanaged, INumberBase<T> => 
        new(this, a, b, output);
    
    public CuTensorPermutationPlan<T> CreatePermutationPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        ReadOnlySpan<Index> axis) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, output, axis);

    public CuTensorReductionPlan<T> CreateReductionPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        HashSet<Index> axis,
        Ops operation,
        bool keepDims = false) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, output, axis, operation, keepDims);

    public CuTensorBroadcastPlan<T> CreateBroadcastPlan<T>(
        AbstractTensor a,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(this, a, output);

    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}