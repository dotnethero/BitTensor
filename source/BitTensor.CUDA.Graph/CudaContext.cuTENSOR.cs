using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;
using System.Numerics;

namespace BitTensor.CUDA.Graph;

using Ops = cutensorOperator_t;

public partial class CudaContext
{
    public CuTensorBinaryPlan<T> CreateUnaryPlan<T>(
        Shape a,
        Shape output,
        Ops unary) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateUnaryPlan<T>(a, output, unary));

    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> => 
        AddResource(cuTENSOR.CreateAggregationPlan<T>(output));

    public CuTensorTernaryPlan<T> CreateAddPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateAddPlan<T>(a, b, output));

    public CuTensorTernaryPlan<T> CreateMultiplyPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateMultiplyPlan<T>(a, b, output));
    
    public CuTensorMatMulPlan<T> CreateMatMulPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateMatMulPlan<T>(a, b, output));

    public CuTensorContractionPlan<T> CreateContractionPlan<T>(
        Shape a,
        Shape b,
        Shape output)
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateContractionPlan<T>(a, b, output));
    
    public CuTensorPermutationPlan<T> CreatePermutationPlan<T>(
        Shape a,
        Shape output,
        ReadOnlySpan<Index> axis) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreatePermutationPlan<T>(a, output, axis));

    public CuTensorReductionPlan<T> CreateReductionPlan<T>(
        Shape a,
        Shape output,
        HashSet<Index> axis,
        Ops operation,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateReductionPlan<T>( a, output, axis, operation, keepDims));

    public CuTensorBroadcastPlan<T> CreateBroadcastPlan<T>(
        Shape a,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        AddResource(cuTENSOR.CreateBroadcastPlan<T>(a, output));
}