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
    
    public void Execute<T>(CudnnExecutionPlan plan, CudnnVariantPack<T> pack) where T : unmanaged, IFloatingPoint<T>
    {
        var status = cuDNN.cudnnBackendExecute(
            this.Handle,
            plan.Descriptor,
            pack.Descriptor);

        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnDestroy(Handle);
    }
}