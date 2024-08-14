using BitTensor.Abstractions;
using System.Numerics;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Wrappers;

using Ops = cutensorOperator_t;

public sealed unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;
    internal readonly List<ICuTensorPlan> Plans = new();

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        Status.EnsureIsSuccess(status);

        Handle = handle;
    }

    public CuTensorBinaryPlan<T> CreateUnaryPlan<T>(
        Shape a,
        Shape output,
        Ops unary) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorBinaryPlan<T>(this, a, output, unary, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD));
    
    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorBinaryPlan<T>(this, output, output, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD));

    public CuTensorTernaryPlan<T> CreateAddPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorTernaryPlan<T>(this, a, b, output, Ops.CUTENSOR_OP_ADD, Ops.CUTENSOR_OP_ADD));

    public CuTensorTernaryPlan<T> CreateMultiplyPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorTernaryPlan<T>(this, a, b, output, Ops.CUTENSOR_OP_MUL, Ops.CUTENSOR_OP_ADD));
    
    public CuTensorMatMulPlan<T> CreateMatMulPlan<T>(
        Shape a,
        Shape b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorMatMulPlan<T>(this, a, b, output));

    public CuTensorContractionPlan<T> CreateContractionPlan<T>(
        Shape a,
        Shape b,
        Shape output)
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorContractionPlan<T>(this, a, b, output));
    
    public CuTensorPermutationPlan<T> CreatePermutationPlan<T>(
        Shape a,
        Shape output,
        ReadOnlySpan<Index> axis) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorPermutationPlan<T>(this, a, output, axis));

    public CuTensorReductionPlan<T> CreateReductionPlan<T>(
        Shape a,
        Shape output,
        HashSet<Index> axis,
        Ops operation,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorReductionPlan<T>(this, a, output, axis, operation, keepDims));

    public CuTensorBroadcastPlan<T> CreateBroadcastPlan<T>(
        Shape a,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Track(new CuTensorBroadcastPlan<T>(this, a, output));
    
    private TPlan Track<TPlan>(TPlan plan) where TPlan : ICuTensorPlan
    {
        Plans.Add(plan);
        return plan;
    }

    public void Dispose()
    {
        var plans = 0;
        foreach (var plan in Plans)
        {
            try
            {
                Console.WriteLine(plan);
                Console.WriteLine(plans);
                plan.Dispose();
                plans++;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
        }
        Console.WriteLine($"{plans} plans disposed");
        cuTENSOR.cutensorDestroy(Handle);
    }
}