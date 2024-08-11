using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorReductionPlan : IDisposable
{
    internal readonly CuTensorDescriptor InputDescriptor;
    internal readonly CuTensorDescriptor OutputDescriptor;
    
    internal readonly CuTensorReduction Reduction;
    internal readonly CuTensorPlan ReductionPlan;
    internal readonly CuTensorWorkspace Workspace;

    internal CuTensorReductionPlan(CuTensorContext context, AbstractTensor input, AbstractTensor output, HashSet<int> axis, cutensorOperator_t op)
    {
        var modes = input.Shape.GetReductionModes(axis);

        InputDescriptor = context.CreateDescriptor(input);
        OutputDescriptor = context.CreateDescriptor(output, modes);

        Reduction = new CuTensorReduction(context, InputDescriptor, OutputDescriptor, OutputDescriptor, op);
        ReductionPlan = Reduction.CreatePlan();
        Workspace = Reduction.CreateWorkspace(ReductionPlan);
    }
    
    public void Execute(CuTensor input, CuTensor output, float alpha = 1f, float beta = 0f) =>
        Reduction.Execute(
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
