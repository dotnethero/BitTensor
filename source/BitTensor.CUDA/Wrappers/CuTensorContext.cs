using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;

namespace BitTensor.CUDA.Wrappers;

public unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        Status.EnsureIsSuccess(status);

        Handle = handle;
    }

    public CuTensorDescriptor CreateDescriptor(AbstractTensor a) => 
        new(this, a);
    
    public CuTensorDescriptor CreateDescriptor(AbstractTensor a, int[] modes) => 
        new(this, a, modes);

    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}