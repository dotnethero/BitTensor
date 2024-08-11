using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    internal CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        Status.EnsureIsSuccess(status);

        Handle = handle;
    }

    internal CuTensorDescriptor CreateDescriptor(AbstractTensor a) => 
        new(this, a);
    
    internal CuTensorDescriptor CreateDescriptor(AbstractTensor a, int[] modes) => 
        new(this, a, modes);

    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}