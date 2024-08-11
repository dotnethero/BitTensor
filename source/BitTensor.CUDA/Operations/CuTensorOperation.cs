using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Operations;

internal unsafe interface ICuTensorOperation : IDisposable
{
    internal CuTensorContext Context { get; }
    internal cutensorOperationDescriptor* Descriptor { get; }

    public CuTensorPlan CreatePlan() => new(this);
    public CuTensorWorkspace CreateWorkspace(CuTensorPlan plan) => new(plan.WorkspaceSize);
}
