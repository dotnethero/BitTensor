using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Plans;

namespace BitTensor.CUDA;

using Ops = cutensorOperator_t;

public partial class CuContext
{
    public CuTensorBinaryPlan CreateUnaryPlan(
        AbstractTensor a,
        AbstractTensor output,
        Ops unary) => 
        new(cuTENSOR, a, output, unary, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    // TODO: axis to broadcast
    public CuTensorBinaryPlan CreateBroadcastPlan(
        AbstractTensor a,
        AbstractTensor output,
        HashSet<int> axis) => 
        new(cuTENSOR, a, output, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorBinaryPlan CreateAggregationPlan(
        AbstractTensor output) => 
        new(cuTENSOR, output, output, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_IDENTITY, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan CreateAddPlan(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) => 
        new(cuTENSOR, a, b, output, Ops.CUTENSOR_OP_ADD, Ops.CUTENSOR_OP_ADD);

    public CuTensorTernaryPlan CreateMultiplyPlan(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) => 
        new(cuTENSOR, a, b, output, Ops.CUTENSOR_OP_MUL, Ops.CUTENSOR_OP_ADD);
    
    public CuTensorMatMulPlan CreateMatMulPlan(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) => 
        new(cuTENSOR, a, b, output);

    public CuTensorContractionPlan CreateContractionPlan(
        AbstractTensor a,
        AbstractTensor b,
        AbstractTensor output) => 
        new(cuTENSOR, a, b, output);
    
    public CuTensorPermutationPlan CreatePermutationPlan(
        AbstractTensor a,
        AbstractTensor output,
        ReadOnlySpan<int> axis) => 
        new(cuTENSOR, a, output, axis);

    public CuTensorReductionPlan CreateSumPlan(
        AbstractTensor a,
        AbstractTensor output) => 
        new(cuTENSOR, a, output, [], Ops.CUTENSOR_OP_ADD);

    public CuTensorReductionPlan CreateSumPlan(
        AbstractTensor a,
        AbstractTensor output,
        HashSet<int> axis) => 
        new(cuTENSOR, a, output, axis, Ops.CUTENSOR_OP_ADD);
}