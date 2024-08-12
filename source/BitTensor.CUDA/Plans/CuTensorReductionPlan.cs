using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Plans;

public sealed class CuTensorReductionPlan<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    internal readonly CuTensorDescriptor<T> InputDescriptor;
    internal readonly CuTensorDescriptor<T> OutputDescriptor;
    internal readonly CuTensorReduction<T> Reduction;
    internal readonly CuTensorPlan ReductionPlan;
    internal readonly CuTensorWorkspace Workspace;

    internal CuTensorReductionPlan(CuTensorContext context, AbstractTensor input, AbstractTensor output, HashSet<Index> axis, cutensorOperator_t op)
    {
        var modes = input.Shape.GetReductionModes(axis);

        InputDescriptor = new(context, input);
        OutputDescriptor = new(context, output, modes);

        Reduction = new(context, InputDescriptor, OutputDescriptor, OutputDescriptor, op);
        ReductionPlan = Reduction.CreatePlan();
        Workspace = Reduction.CreateWorkspace(ReductionPlan);
    }
    
    public void Execute(IDeviceArray<T> input, IDeviceArray<T> output, float alpha = 1f, float beta = 0f) =>
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
