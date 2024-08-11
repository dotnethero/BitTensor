using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA;

using Ops = cutensorOperator_t;

public partial class CuContext
{
    public CuTensorBinaryPlan<T> CreateUnaryPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        Ops unary) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, output, unary, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, output, output, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan<T> CreateAddPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, b, output, Ops.CUTENSOR_OP_ADD, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan<T> CreateMultiplyPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, b, output, Ops.CUTENSOR_OP_MUL, Ops.CUTENSOR_OP_ADD);
    
    public CuTensorMatMulPlan<T> CreateMatMulPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, b, output);

    public CuTensorContractionPlan<T> CreateContractionPlan<T>(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output)
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, b, output);
    
    public CuTensorPermutationPlan<T> CreatePermutationPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        ReadOnlySpan<int> axis) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, output, axis);

    public CuTensorReductionPlan<T> CreateSumPlan<T>(
        AbstractTensor a,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, output, [], Ops.CUTENSOR_OP_ADD);

    public CuTensorReductionPlan<T> CreateSumPlan<T>(
        AbstractTensor a,
        AbstractTensor output,
        HashSet<int> axis) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, output, axis, Ops.CUTENSOR_OP_ADD);
    
    public CuTensorBroadcastPlan<T> CreateBroadcastPlan<T>(
        AbstractTensor a,
        AbstractTensor output) 
        where T : unmanaged, INumberBase<T> => 
        new(cuTENSOR, a, output);
}