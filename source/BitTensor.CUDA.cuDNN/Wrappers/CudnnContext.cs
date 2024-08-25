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

    public void Dispose()
    {
        cuDNN.cudnnDestroy(Handle);
    }
}