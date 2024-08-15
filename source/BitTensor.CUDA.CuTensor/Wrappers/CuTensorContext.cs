using BitTensor.Abstractions;
using System.Numerics;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA.Wrappers;

using OpCode = cutensorOperator_t;

public sealed unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;
    internal readonly HashSet<ICuTensorPlan> Plans = [];

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        Status.EnsureIsSuccess(status);

        Handle = handle;
    }
    
    public CuTensorBroadcastPlan<T> CreateBroadcastPlan<T>(
        Operand a,
        Operand output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorBroadcastPlan<T>(this, a, output));

    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        Shape output)
        where T : unmanaged, IFloatingPoint<T> =>
        CreateAggregationPlan<T>(output, output);

    public CuTensorBinaryPlan<T> CreateAggregationPlan<T>(
        Operand a,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorBinaryPlan<T>(this, a, output, OpCode.CUTENSOR_OP_ADD));

    public CuTensorTernaryPlan<T> CreateAddPlan<T>(
        Operand a,
        Operand b,
        Operand output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorTernaryPlan<T>(this, a, b, output, OpCode.CUTENSOR_OP_ADD, OpCode.CUTENSOR_OP_ADD));

    public CuTensorTernaryPlan<T> CreateMultiplyPlan<T>(
        Operand a,
        Operand b,
        Operand output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorTernaryPlan<T>(this, a, b, output, OpCode.CUTENSOR_OP_MUL, OpCode.CUTENSOR_OP_ADD));
    
    public CuTensorMatMulPlan<T> CreateMatMulPlan<T>(
        Operand a,
        Operand b,
        Shape output) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorMatMulPlan<T>(this, a, b, output));

    public CuTensorContractionPlan<T> CreateContractionPlan<T>(
        Operand a,
        Operand b,
        Operand output)
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorContractionPlan<T>(this, a, b, output));
    
    public CuTensorPermutationPlan<T> CreatePermutationPlan<T>(
        Operand a,
        Shape output,
        ReadOnlySpan<Index> axis) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorPermutationPlan<T>(this, a, output, axis));

    public CuTensorReductionPlan<T> CreateReductionPlan<T>(
        Operand a,
        Operand output,
        HashSet<Index> axis,
        OpCode operation,
        bool keepDims = false) 
        where T : unmanaged, IFloatingPoint<T> =>
        Add(new CuTensorReductionPlan<T>(this, a, output, axis, operation, keepDims));

    private TPlan Add<TPlan>(TPlan plan) where TPlan : ICuTensorPlan
    {
        Plans.Add(plan);
        return plan;
    }
    
    private void FreeResources()
    {
        var plans = 0;
        foreach (var plan in Plans)
        {
            plans++;
            plan.Dispose();
        }
        Plans.Clear();
        Console.Error.WriteLine($"{plans} operation plans disposed");
    }

    public void Dispose()
    {
        FreeResources();
        cuTENSOR.cutensorDestroy(Handle);
    }
}