using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudnnContext: IDisposable
{
    public cudnnContext* Handle { get; }

    public CudnnContext()
    {
        cudnnContext* handle;
        var status = cuDNN.cudnnCreate(&handle);
        Status.EnsureIsSuccess(status);
        Handle = handle;
    }
    
    public void ExecutePlan<T>(CudnnExecutionPlan plan, CudnnVariantPack<T> pack) where T : unmanaged, IFloatingPoint<T>
    {
        var status = cuDNN.cudnnBackendExecute(
            this.Handle,
            plan.Descriptor,
            pack.Descriptor);

        Status.EnsureIsSuccess(status);
    }

    public void ExecuteGraph<T>(CudnnGraph graph, CudnnVariantPack<T> pack) where T : unmanaged, IFloatingPoint<T>
    {
        using var heuristics = new CudnnEngineHeuristics(graph);
        using var config = heuristics.GetConfiguration();
        using var engine = new CudnnEngine(graph, globalIndex: 0);
        using var plan = new CudnnExecutionPlan(this, config);
        ExecutePlan(plan, pack);
    }

    public void Dispose()
    {
        cuDNN.cudnnDestroy(Handle);
    }
}