﻿using BitTensor.Abstractions;
using BitTensor.CUDA.ComputeOnly.Wrappers;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Plans;

internal sealed class CuTensorSumPlan(
    CuTensorContext context,
    CuTensor input,
    CuTensor output,
    HashSet<int> axis) : 
    CuTensorReductionPlan(context, input, output, axis, cutensorOperator_t.CUTENSOR_OP_ADD);

internal abstract class CuTensorReductionPlan : IDisposable
{
    internal readonly CuTensorDescriptor InputDescriptor;
    internal readonly CuTensorDescriptor OutputDescriptor;
    
    internal readonly CuTensorReduction Reduction;
    internal readonly CuTensorPlan ReductionPlan;
    internal readonly CuTensorWorkspace Workspace;

    protected CuTensorReductionPlan(CuTensorContext context, CuTensor input, CuTensor output, HashSet<int> axis, cutensorOperator_t op)
    {
        var modes = input.Shape.GetReductionModes(axis);

        InputDescriptor = context.CreateDescriptor(input);
        OutputDescriptor = context.CreateDescriptor(output, modes);

        Reduction = new CuTensorReduction(context, InputDescriptor, OutputDescriptor, OutputDescriptor, op);
        ReductionPlan = Reduction.CreatePlan();
        Workspace = Reduction.CreateWorkspace(ReductionPlan);
    }
    
    public void Execute(CuTensor input, CuTensor output, float alpha = 1f, float beta = 0f) =>
        Reduction.ExecuteByPlan(
            ReductionPlan,
            Workspace,
            input,
            output,
            output,
            alpha,
            beta);

    public void Dispose()
    {
        ReductionPlan.Dispose();
        Workspace.Dispose();
        Reduction.Dispose();
        OutputDescriptor.Dispose();
        InputDescriptor.Dispose();
    }
}
